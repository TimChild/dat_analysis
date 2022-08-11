from filelock import FileLock
import h5py
import time
from queue import Queue
from deprecation import deprecated
import logging
import threading
from typing import Dict, Tuple, List, Optional, Callable
import os
import tempfile
from .core_util import TEMPDIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HDF_GLOBAL_LOCK_PATH = os.path.join(TEMPDIR, 'hdf_global_lock.lock')

READ_MODES = ['r']
WRITE_MODES = ['r+', 'w']


class HDF:
    def __init__(self, hdf_path: str):
        self._hdf_path = hdf_path

    @property
    def hdf_read(self):
        """Explicitly open hdf for reading

        Examples:
            with dat.hdf_read as f:
                data = f['data'][:]
        """
        return HDFFileHandler(self._hdf_path, 'r')  # with self.hdf_read as f: ...

    @property
    def hdf_write(self):
        """
        Explicitly open hdf for writing

        Examples:
            with dat.hdf_write as f:
                f['data'] = np.array([1, 2, 3])
        """
        return HDFFileHandler(self._hdf_path, 'r+')  # with self.hdf_write as f: ...


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
        # self.file.close()  # Ensure old file is closed
        # TODO: Any better way to ensure old file is closed?
        if file:
            self._id = file.id
        else:
            raise RuntimeError(f'Previous file already closed so cannot get file.id from it')


class FileQueue:
    def __init__(self, filepath: str):
        assert isinstance(filepath, str)
        self.filepath = filepath
        self.read_queue = {}
        self.write_queue = []
        self.working = {}
        self.trigger = threading.Condition()  # Also acts like a thread lock
        self.writing_thread = None
        self.worker: threading.Thread = None
        self.filelock = GlobalLock(os.path.normpath(filepath) + '.lock')  # Prevent any other threads/processes
        self.threadlock = threading.Lock()  # To make sure thread gets a chance to set manager_waiting back to False
        self.manager_waiting = False
        self.kill_flag = False

    def kill(self, possible_file: [None, h5py.File]):
        """Kill the filequeue threads, something has gone wrong"""
        if possible_file is not None:
            possible_file.close()
        self.kill_flag = True  # ends worker processes

        # Release anything waiting
        for k in self.read_queue:
            logger.warning(f'Killing FileQueue: Releasing read threads for thread_id: {k}')
            for event in self.read_queue[k]:
                event.set()
        for id_, event in self.write_queue:
            logger.warning(f'Killing FileQueue: Releasing write threads for thread_id: {id_}')
            event.set()

        # Reset all other variables because this could be used again by HDFFileHandler
        # (that keeps track of existing FileQueues)
        # self.read_queue = {}
        # self.write_queue = []
        # self.working = {}
        # self.writing_thread = None
        # self.worker = None
        # self.manager_waiting = False
        # self._kill_flag = False
        logger.error(f'Killed FileQueue for {self.filepath}')

    def ensure_worker_alive(self):
        def worker_manager():
            """Starts the worker thread, and waits for it to finish to release filelock without blocking rest of code"""

            def start_worker():
                self.worker = threading.Thread(target=self.worker_job)
                self.worker.start()

            with self.filelock:
                if not self.worker or isinstance(self.worker, threading.Thread) and self.worker.is_alive() is False:
                    logger.debug(f'Starting a new worker, will wait for it to finish')
                    start_worker()
                    while True:
                        self.worker.join(1)
                        if self.kill_flag:
                            self.manager_waiting = False
                            logger.error(f'Kill flag received, ending worker_manager regardless of worker state')
                            return False
                        with self.trigger:
                            if not self.worker.is_alive():
                                if self.write_queue or self.read_queue:
                                    logger.debug(
                                        f'Previous worker finished, but already new queues, starting new worker')
                                    start_worker()  # New additions after worker ended, need to start again
                                else:
                                    self.manager_waiting = False
                                    break  # I.e. if the join was successful, break waiting loop

            logger.debug('Worker finished, exiting')
            return True

        if not self.worker or \
                isinstance(self.worker, threading.Thread) and \
                not self.worker.is_alive() and \
                not self.manager_waiting:
            self.manager_waiting = True
            logger.debug(f'Starting a new worker_manager')
            t = threading.Thread(target=worker_manager)
            t.start()
        else:
            worker_still_required = True

        # Worker or manager must be ready when leaving here

    def add_to_queue(self, thread_id, mode) -> threading.Event:
        if self.kill_flag:
            raise RuntimeError(f'FileQueue previously killed, should not be used again.')
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
            self.ensure_worker_alive()
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
                if file:
                    file.close()
                self.writing_thread = None
                return True
        return False

    def worker_job(self):
        """Decides when each thread is allowed to access the file"""
        while True:
            with self.trigger:
                logger.debug(f'Worker doing stuff')

                # If FileQueue needs to be killed
                if self.kill_flag:
                    logger.error(f'kill_flag received, stopping worker')
                    return False

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
                    logger.debug(f'Worker finished all work, ending')
                    return True  # Kill the worker (will be created again if anything added to queue)
                else:
                    logger.debug(f'No action for {self.filepath}:\n'
                                 f'\tWorking: {self.working}\n'
                                 f'\tRead_queue: {self.read_queue}\n'
                                 f'\tWrite_queue: {self.write_queue}')

                logger.debug(f'Worker finished waiting for trigger')
                self.trigger.wait(timeout=5)  # At least try once, then wait for more notifications


def _wait_until_free(filepath, timeout=20):
    from .hdf_util import wait_until_file_free
    return wait_until_file_free(filepath, timeout=timeout)


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

    def __init__(self, filepath: str, filemode: str, open_file_timeout=20):
        """

        Args:
            filepath (): Path to file
            filemode (): mode to open file in  (e.g. 'r' for read, 'r+' for write on existing file,  'w' for new, etc)
        """
        self._filepath = filepath
        self._filemode = filemode
        self._thread_id = threading.get_ident()
        self._open_file_timeout = open_file_timeout
        self._in_use = False

    def __enter__(self):
        """
        For context manager
        """
        if self._in_use:
            raise RuntimeError(f'Cannot nest a single handler, create a new instance instead')

        # Get the filequeue manager
        filequeue = self._get_file_queue(self._filepath)

        # check if able to use filepath
        allowed = self.wait_until_available(filequeue, self._filemode, timeout=30)  # Seems unlikely the queue should need to be longer than 30s
        if not allowed:
            filequeue.kill(possible_file=self.get_file(self._filepath))
            raise RuntimeError(f'Error while waiting for file to become available internally ('
                               f'i.e. waiting on FileQueue for more than 60s or FileQueue killed).')

        # save current status of filepath to return state at end of context
        file = self.get_file(self._filepath)
        self._previous_state = dict(
            mode=file.mode if file else None,
        )

        # set filepath to desired state and return
        try:
            file = self.set_file_state(self._filepath, self._filemode, timeout=self._open_file_timeout)  # This timeout is if an external process holds the file open
            self._in_use = True
        except TimeoutError as e:
            filequeue.kill(possible_file=self.get_file(self._filepath))
            raise TimeoutError(f'Failed to access {self._filepath} due to external process having file open')
        return file

    def __exit__(self, exc_type, exc_val, exc_tb):
        """For context manager"""
        fq = self._get_file_queue(self._filepath, return_killed=True)  # Don't replace a killed FileQueue here
        try:
            # return filepath to previous state or close as necessary
            if not fq.kill_flag:
                closed = fq.finish(self._thread_id, self.get_file(self._filepath))
                if not closed and not exc_type:
                    self.set_file_state(
                        filepath=self._filepath,
                        filemode=self._previous_state.get('mode'),
                        timeout=self._open_file_timeout
                    )
        except Exception as e:
            fq.kill(self.get_file(self._filepath))
            raise e
        finally:
            self._in_use = False

    def wait_until_available(self, filequeue, filemode, timeout):
        # add to queue for read or write
        # only allow to pass if only others are reading and no write in queue
        allowed_to_pass = filequeue.add_to_queue(self._thread_id, filemode)
        allowed = allowed_to_pass.wait(timeout=timeout)
        if filequeue.kill_flag:
            logger.debug(f'FileQueue.kill_flag set, not not allowed to use file')
            allowed = False
        return allowed

    def _get_file_queue(self, filepath, return_killed=False):
        fq = self._file_queues.get(filepath, None)
        if fq is None or fq.kill_flag and return_killed is False:
            with self._global_lock:  # May need to create new
                fq = self._file_queues.get(filepath, None)  # May have been created by another waiting thread already
                if not fq or fq.kill_flag and return_killed is False:  # Need to create or make new FileQueue
                    fq = FileQueue(self._filepath)
                    self._file_queues[filepath] = fq
        return fq

    def get_file(self, filepath):
        if filepath not in self._files:
            self._files[filepath] = None
        return self._files[filepath]

    def set_file_state(self, filepath, filemode, timeout):
        """Opens file if necessary. Changes file state from read -> write if necessary.
        Does NOT change from write -> read, just leaves in write mode
            -- This is so that a writing thread can finish
            whatever it is doing (including read sections) without being interrupted"""
        file = self.get_file(filepath)
        if not file:
            if filemode != 'w' and not os.path.exists(filepath):
                raise FileNotFoundError(f'No file found at {filepath}. HDF file must already exist')
            elif filemode in WRITE_MODES:  # Not OK to use an Open HDF file in Write mode (i.e. if open in HDF Viewer)
                _wait_until_free(filepath, timeout=timeout)

            file = FlexibleFile(filepath, filemode)  # an h5py File where the .id can be switched to point to new HDF
            self._files[filepath] = file

        if filemode and file.mode != filemode:
            # Always switch read -> write, never switch write -> read
            if filemode in WRITE_MODES and file.mode in READ_MODES:
                file.close()
                _wait_until_free(filepath,
                                 timeout=timeout)  # Not OK to use an Open HDF file in Write mode (i.e. if open in HDF Viewer)
                new_file = h5py.File(filepath, filemode)
                file.switch_file(new_file)  # So file now points to newly opened file

        return file


@deprecated(deprecated_in='3.0.0', details='This was the old HDFFileHandler which is inferior to the new HDFFileHandler')
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
