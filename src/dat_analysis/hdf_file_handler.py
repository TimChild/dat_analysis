from filelock import FileLock
import h5py
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


class HDFFileHandler:
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
        self._acquire()
        try:
            self._file = h5py.File(self._filepath, self._filemode)
            if self._internal_path is not None:
                group = self._file[self._internal_path]
                return group
            else:
                return self._file
        except Exception as e:
            self._release()
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb):
        """For context manager"""
        try:
            self._file.close()
        except Exception as e:
            logger.error(f'Error closing {self._filepath}. \nException: {e}')
        finally:
            self._release()

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
