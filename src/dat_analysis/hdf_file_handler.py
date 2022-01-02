from filelock import FileLock
import h5py
import logging
import os
import threading
import time
from typing import Dict, Tuple



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
    Allow a single thread in a single process to work with a file
    """
    _global_lock = GlobalLock('global_filelock.lock')  # A lock that only one thread/process can hold
    _file_locks: Dict[str, Tuple[int, GlobalLock]] = {}  # Lock for each file that is open

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
        self._file.close()
        if self._acquired_lock:
            self._release_file_lock()

    def __enter__(self):
        self._acquire()
        try:
            self._file = h5py.File(self._filepath, self._filemode)
        except PermissionError:
            self._release()
        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._release()

    def _file_free(self) -> bool:
        """Works on Windows only... Checks if file is open by any process"""
        try:
            os.rename(self._filepath, self._filepath)
            # logging.info('File free')
            return True
        except PermissionError as e:
            # logging.info('File not accessible')
            return False

    def _wait_until_free(self, timeout=30):
        start = time.time()
        while time.time() - start < timeout:
            while not self._file_free():
                time.sleep(0.1)
            if self._file_free():
                logging.info('file is free')
                return
        raise TimeoutError(f'File not accessible within timeout of {timeout}s')
