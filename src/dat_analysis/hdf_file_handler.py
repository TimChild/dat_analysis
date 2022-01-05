from filelock import FileLock
import h5py
import logging
import os
import threading
import time
from typing import Dict, Tuple, List



class GlobalLock:
    """Both threading and process lock"""
    def __init__(self, lock_filepath: str):
        self.filelock = FileLock(lock_filepath)
        self.threadlock = threading.Lock()

    def acquire(self):
        self.threadlock.acquire()
        self.filelock.acquire()

    def release(self):
        self.filelock.release()
        self.threadlock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class HDFFileHandler:
    """
    Allow a single thread in a single process to work with an HDF file (which can only be opened once)
    """
    _global_lock = GlobalLock('global_filelock.lock')  # A lock that only one thread/process can hold
    _file_locks: Dict[str, Tuple[int, GlobalLock]] = {}  # Lock for each file that is open
    _open_file_modes: Dict[str, Tuple[h5py.File, List[str]]] = {}  # Keep track of requested filemode

    def __init__(self, filepath: str, filemode: str):
        """

        Args:
            filepath (): Path to file
            filemode (): mode to open file in
        """
        self._filepath = filepath
        self._filemode = filemode
        self._acquired_lock = False
        self.thread_id = threading.get_ident()

        self._file = None

    def _init_lock(self):
        """Make sure there is a FileLock existing for the file"""
        with self._global_lock:
            if self._filepath not in self._file_locks:
                self._file_locks[self._filepath] = (-1, GlobalLock(self._filepath+'.lock'))

    def _acquire_file_lock(self):
        with self._global_lock:
            lock = self._file_locks[self._filepath][1]
        lock.acquire()
        logging.info(f'lock on {self._filepath} acquired')
        self._file_locks[self._filepath] = (self.thread_id, lock)
        self._acquired_lock = True

    def _release_file_lock(self):
        with self._global_lock:
            logging.info(f'global lock acquired')
            lock = self._file_locks[self._filepath][1]
            self._file_locks[self._filepath] = (-1, lock)
            lock.release()
            logging.info(f'lock on {self._filepath} released')

    def _acquire(self):
        """Acquire rights to open file"""
        self._init_lock()
        if self._file_locks[self._filepath][0] != self.thread_id:  # If not already the thread with the file lock
            self._acquire_file_lock()
            self._wait_until_free()  # Wait until the file is free from any other processes after acquiring file lock

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
        except OSError:
            self._release()
        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb):
        """For context manager"""
        self._file.close()
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
            return file
        except Exception as e:
            self._release()
            raise e

    def previous(self):
        """
        Get the previous state of the file before the last .new() was called (or closes the file .new() was the first access)
        """
        try:
            file, modes = self._open_file_modes[self._filepath]
            if len(modes) == 1:  # Last release, just need to close file and remove entry from list
                file.close()
                self._open_file_modes.pop(self._filepath)
                file = None
            else:
                current_mode = modes.pop()
                last_mode = modes[-1]
                if current_mode != last_mode:
                    file.close()
                    file = h5py.File(self._filepath, last_mode)
                self._open_file_modes[self._filepath] = (file, modes)
        finally:
            self._release()
        return file

    def _wait_until_free(self):
        from .hdf_util import wait_until_file_free
        return wait_until_file_free(self._filepath)
