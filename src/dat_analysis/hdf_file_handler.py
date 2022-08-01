from filelock import FileLock
import h5py
import time
from queue import Queue
import logging
import threading
from typing import Dict, Tuple, List, Optional
import os
import tempfile

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

tempdir = os.path.join(tempfile.gettempdir(), 'dat_analysis')
os.makedirs(tempdir, exist_ok=True)
HDF_GLOBAL_LOCK_PATH = os.path.join(tempdir, 'hdf_global_lock.lock')

READ_MODES = ['r']
WRITE_MODES = ['r+', 'w']


class GlobalLock:
    """Both threading and process lock"""

    def __init__(self, lock_filepath: str):
        self.filelock = FileLock(lock_filepath)
        self.threadlock = threading.Lock()
        self._filepath = lock_filepath

    def acquire(self):
        # logger.debug(f'beginning global lock acquire for {self._filepath}')
        self.threadlock.acquire()
        self.filelock.acquire()
        # logger.debug(f'global lock acquired for {self._filepath}')

    def release(self):
        # logger.debug(f'beginning global lock release for {self._filepath}')
        self.filelock.release()
        self.threadlock.release()
        # logger.debug(f'global lock released for {self._filepath}')

    def __enter__(self):
        self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class FlexibleFile(h5py.File):
    def switch_file(self, file: h5py.File):
        self._id = file.id


class FileQueue:
    def __init__(self, filepath):
        self.filepath = filepath
        self.read_queue = {}
        self.write_queue = []
        self.working = {}
        self.trigger = threading.Condition()  # Need to add a trigger queue or something
        self.writing_thread = None
        self.worker: threading.Thread = None

    def add_to_queue(self, thread_id, mode) -> threading.Event:
        event = threading.Event()
        with self.trigger:
            if mode in READ_MODES:
                queue = self.read_queue
                queue[thread_id] = queue.get(thread_id, [])
                queue[thread_id].append(event)
            elif mode in WRITE_MODES:
                queue = self.write_queue
                queue.append((thread_id, event))
            else:
                raise ValueError(f'{mode} not recognized')

            if not self.worker or isinstance(self.worker, threading.Thread) and self.worker.is_alive() is False:
                self.worker = threading.Thread(target=self.worker_job)
                self.worker.start()
            self.trigger.notify()
        return event

    def finish(self, thread_id, file: h5py.File):
        with self.trigger:
            self.working[thread_id].pop()  # pop a single thread use
            if not self.working[thread_id]:
                self.working.pop(thread_id)  # thread no longer using
            closed = self.close_if_last_access(file)
            self.trigger.notify()
        return closed

    def close_if_last_access(self, file: h5py.File):
        """Returns True if only a single access to the file by a single thread"""
        with self.trigger:
            if not self.working:
                file.close()
                self.writing_thread = None
                return True
        return False

    def worker_job(self):
        while True:
            with self.trigger:
                # Allow all reads to start if no write queue and no writing thread
                if not self.write_queue and not self.writing_thread and self.read_queue:
                    for thread_id, events in self.read_queue.items():
                        self.working[thread_id] = self.working.get(thread_id, [])
                        for e in reversed(events):  # FIFO
                            self.working[thread_id].append(e)
                            e.set()
                    self.read_queue = {}

                # If a write is pending
                elif self.write_queue:

                    # If already writing, check if any more write requests for the same thread exist
                    if self.writing_thread:
                        for i, (thread_id, e) in enumerate(self.write_queue):
                            if thread_id == self.writing_thread:
                                self.write_queue.pop(i)  # Remove that entry from queue and start it, then continue on
                                self.working[thread_id].append(e)
                                e.set()
                                break
                    else:
                        thread_id, e = self.write_queue[0]
                        # Allow a write if no working threads or working threads in same thread
                        if not self.working or len(self.working.keys()) == 1 and thread_id in self.working.keys():
                            self.working[thread_id] = self.working.get(thread_id, [])
                            self.writing_thread = thread_id
                            self.working[thread_id].append(e)
                            self.write_queue.pop(0)  # Dealt with first in queue, so remove now
                            e.set()

                # Allow reads if in same thread as writing thread
                elif self.read_queue and self.writing_thread:
                    if self.writing_thread in self.read_queue:
                        events = self.read_queue.pop(self.writing_thread)
                        self.working[thread_id] = self.working.get(thread_id, [])
                        for e in reversed(events):  # FIFO
                            self.working[thread_id].append(e)
                            e.set()

                elif not self.working and not self.read_queue and not self.write_queue:
                    return True  # Kill the worker (will be created again if anything added to queue)
                else:
                    logger.debug(f'No action for {self.filepath}')
                self.trigger.wait(1)  # At least try once, then wait for more notifications


class HDFFileHandler:
    """
    Allows multiple threads to read a file, or a single thread/process to write to a file

    Examples:
        with HDFFileHandler('test.h5', 'r') as f:
            data = f.get('data')[:]


        # Note: Group ids are lost if switching from read -> write intent
        with HDFFileHandler('test.h5', 'r') as f:
            g = f.get('group')
            with HDFFileHandler('test.h5', 'r+') as f2:
                f['group'].attrs['a'] = 1
            #  v = g.attrs['a']  # <<< This is invalid
            g = f.get('group')
            v = g.attrs['a']  # <<< OK if g is defined again

    """
    _global_lock = GlobalLock(HDF_GLOBAL_LOCK_PATH)  # A lock that only one thread/process can hold
    _files = dict()
    _file_queues = dict()

    def __init__(self, filepath: str, filemode: str):
        """

        Args:
            filepath (): Path to file
            filemode (): mode to open file in  (e.g. 'r' for read, 'r+' for write on existing file,  'w' for new, etc)
        """
        self._filepath = filepath
        self._filemode = filemode
        self._thread_id = threading.get_ident()
        self._in_use = False

    def __enter__(self):
        """
        For context manager
        """
        if self._in_use:
            raise RuntimeError(f'Cannot nest a single handler, create a new instance instead')

        # check if able to use filepath
        self.wait_until_available(self._filepath, self._filemode)

        # save current status of filepath to return state at end of context
        file = self.get_file(self._filepath)
        self._previous_state = dict(
            open=bool(file),
            mode=file.mode if file else None,
        )

        # set filepath to desired state and return
        file = self.set_file_state(self._filepath, self._filemode)
        self._in_use = True
        return file

    def __exit__(self, exc_type, exc_val, exc_tb):
        """For context manager"""
        try:
            # return filepath to previous state or close as necessary
            fq = self._get_file_queue(self._filepath)
            closed = fq.finish(self._thread_id, self.get_file(self._filepath))
            if not closed and not exc_type:
                self.set_file_state(
                    filepath=self._filepath,
                    filemode=self._previous_state.get('mode'),
                )
        except Exception as e:
            file = self.get_file(self._filepath)
            if file:
                file.close()
            raise e
        finally:
            self._in_use = False

    def wait_until_available(self, filepath, filemode):
        # add to queue for read or write
        # only allow to pass if only others are reading and no write in queue
        fq = self._get_file_queue(filepath)
        allowed_to_pass = fq.add_to_queue(self._thread_id, filemode)
        allowed_to_pass.wait()

    def _get_file_queue(self, filepath):
        if filepath in self._file_queues:
            fq = self._file_queues[filepath]
        else:
            with self._global_lock:
                if filepath in self._file_queues:
                    fq = self._file_queues[filepath]
                else:
                    fq = FileQueue(self._filepath)
                    self._file_queues[filepath] = fq
        return fq

    def get_file(self, filepath):
        if filepath not in self._files:
            self._files[filepath] = None
        return self._files[filepath]

    def set_file_state(self, filepath, filemode):
        file = self.get_file(filepath)
        if not file:
            file = FlexibleFile(filepath, filemode)
            self._files[filepath] = file

        if filemode and file.mode != filemode:
            # Always switch read -> write, never switch write -> read
            if filemode in WRITE_MODES and file.mode in READ_MODES:
                file.close()
                new_file = h5py.File(filepath, filemode)
                file.switch_file(new_file)  # So file now points to newly opened file

        return file

    def _wait_until_free(self):
        from .hdf_util import wait_until_file_free
        return wait_until_file_free(self._filepath, timeout=20)


class Old_HDFFileHandler:
    """
    Allow a single thread in a single process to work with an HDF file (which can only be opened once)

    Examples:
        with HDFFileHandler('test.h5', 'r') as f:
            data = f.get('data')[:]

        OR

        fh = HDFFileHandler('test.h5', 'r')
        fh.new()  # Get a new access (a single thread can enter multiple times even with different mode)
        dataset = f.get('data')
        do_something_with_dataset(dataset)
        fh.previous()  # Return to previous state (reverts state of file to previous state e.g. 'r', 'r+', closed etc)


    """
    _global_lock = GlobalLock(HDF_GLOBAL_LOCK_PATH)  # A lock that only one thread/process can hold
    _file_locks: Dict[str, Tuple[int, GlobalLock]] = {}  # Lock for each file that is open
    _open_file_modes: Dict[str, Tuple[h5py.File, List[str]]] = {}  # Keep track of requested filemode

    _managers = dict()

    def __new__(cls, *args, **kwargs):
        with cls._global_lock:
            thread_id = threading.get_ident()
            if thread_id not in cls._managers:
                cls._managers[thread_id] = [super().__new__(cls), 1]  # Create manager, and 1 instance
            else:
                cls._managers[thread_id][1] += 1  # Increment number of instances of this manager
            return cls._managers[thread_id][0]

    def __del__(self):
        """Tidy up the class store of managers"""
        with self._global_lock:
            thread_id = threading.get_ident()
            self._managers[thread_id][1] -= 1
            if self._managers[thread_id][1] == 0:  # If last existing instance of this manager, remove from class dict
                self._managers.pop(thread_id)

    def __init__(self, filepath: str, filemode: str, internal_path: Optional[str] = None):
        """

        Args:
            filepath (): Path to file
            filemode (): mode to open file in  (e.g. 'r' for read, 'r+' for write on existing file,  'w' for new, etc)
        """
        self._filepath = filepath
        self._filemode = filemode
        self._internal_path = internal_path  # I.e. to a group or dataset inside the HDF (e.g. '/top_group/sub_group')
        self._acquired_lock = False
        self.thread_id = threading.get_ident()
        self._file = None

    def _init_lock(self):
        """Make sure there is a FileLock existing for the file"""
        with self._global_lock:  # Prevent race condition on filelock creation
            if self._filepath not in self._file_locks:
                self._file_locks[self._filepath] = (-1, GlobalLock(self._filepath + '.lock'))

    def _acquire_file_lock(self):
        logger.debug(f'attempting lock on {self._filepath}')
        with self._global_lock:  # Avoid reading from self._file_locks while it is being modified
            lock = self._file_locks[self._filepath][1]
        logger.debug(f'waiting to lock {self._filepath}')
        lock.acquire()
        logger.debug(f'lock on {self._filepath} acquired')
        with self._global_lock:  # Avoid writing to self._file_locks while it is being modified
            self._file_locks[self._filepath] = (self.thread_id, lock)
        self._acquired_lock = True

    def _release_file_lock(self):
        with self._global_lock:
            lock = self._file_locks[self._filepath][1]
            self._file_locks[self._filepath] = (-1, lock)
            lock.release()
            logger.debug(f'lock on {self._filepath} released')

    def _acquire(self):
        """Acquire rights to open file"""
        self._init_lock()
        if self._file_locks[self._filepath][0] != self.thread_id:  # If not already the thread with the file lock
            logger.debug(f'require lock to open {self._filepath}')
            self._acquire_file_lock()
            try:
                self._wait_until_free()  # Wait until the file is free from any other processes after acquiring file lock
            except TimeoutError as e:  # If file does not become free, release lock before raising error
                self._release_file_lock()
                raise e

    def _release(self):
        """Release rights to open file"""
        if self._acquired_lock:
            self._release_file_lock()

    def __enter__(self):
        """
        For context manager
        Note: Cannot use nested access with context manager for h5py Files. Use .new() and .previous() instead
        """
        # self._acquire()
        # try:
        #     if not self._file:
        #         self._file = h5py.File(self._filepath, self._filemode)
        #         self._opened_file = True
        #     if self._internal_path is not None:
        #         group = self._file[self._internal_path]
        #         return group
        #     else:
        #         return self._file
        # except Exception as e:
        #     self._release()
        #     raise e
        return self.new()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """For context manager"""
        # try:
        #     self._file.close()
        # except Exception as e:
        #     logger.error(f'Error closing {self._filepath}. \nException: {e}')
        # finally:
        #     self._release()
        self.previous()

    def new(self):
        """Get a new access to the file
        Note: Remember to call .previous() after use! (use a try - finally to ensure it is always called)
        """
        self._acquire()
        try:
            if not self._filepath in self._open_file_modes:
                file = h5py.File(self._filepath, self._filemode)
                self._open_file_modes[self._filepath] = (file, [self._filemode])
            else:
                file, modes = self._open_file_modes[self._filepath]
                if self._filemode != modes[-1]:  # Need to close and reopen in new mode
                    file.close()
                    file = h5py.File(self._filepath, self._filemode)
                modes.append(self._filemode)
                self._open_file_modes[self._filepath] = (file, modes)
            if not file:
                raise RuntimeError(f'File is not open?!')
            if self._internal_path is not None:
                group = file[self._internal_path]
                return group
            else:
                return file
        except Exception as e:
            self._release()  # Only called if failed to get a new access to HDF
            raise e

    def previous(self):
        """
        Get the previous state of the file before the last .new() was called (or closes the file if .new() was the first access)
        """
        # Note: filelock should already be acquired by the .new()
        try:
            file, modes = self._open_file_modes[self._filepath]
            if len(modes) == 1:  # Last release, just need to close file and remove entry from list
                file.close()
                self._open_file_modes.pop(self._filepath)
                file = None
            else:
                current_mode = modes.pop()
                last_mode = modes[-1]
                if not file:  # If file is closed
                    logger.warning(f'File at {self._filepath} was found closed before it should have been, reopening')
                    file = h5py.File(self._filepath, last_mode)
                elif current_mode != last_mode:
                    file.close()
                    file = h5py.File(self._filepath, last_mode)
                self._open_file_modes[self._filepath] = (file, modes)
        finally:
            self._release()  # Definitely want to release the filelock whether successful or an error raised.
        return file

    def _wait_until_free(self):
        from .hdf_util import wait_until_file_free
        return wait_until_file_free(self._filepath, timeout=20)
