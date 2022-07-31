from unittest import TestCase
import os
import time
import h5py
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dat_analysis.hdf_file_handler import HDFFileHandler

# Not permanent, free to be deleted in tests etc
output_path = os.path.normpath(os.path.join(__file__, '../Outputs/hdf_file_manager/'))
os.makedirs(output_path, exist_ok=True)

fp1 = os.path.join(output_path, 'f1.h5')


def _test_read_write_read(v, start_delay):
    handler = HDFFileHandler(fp1, 'r+')
    time.sleep(start_delay)
    with handler as f:
        f.attrs['test'] = v
        time.sleep(1)
        read = f.attrs['test']
    return read


class TestHDFFileHandler(TestCase):

    def setUp(self) -> None:
        if os.path.exists(fp1):
            os.remove(fp1)
        with h5py.File(fp1, 'w') as f:
            f.attrs['a'] = 'a'
            f['data'] = np.array([1, 2, 3])
            f.require_group('group_1')

    def tearDown(self) -> None:
        pass

    def test_basic_read_context(self):
        handler = HDFFileHandler(fp1, 'r')
        with handler as f:
            v = f.attrs['a']
            d = f['data'][:]

        self.assertEqual('a', v)
        self.assertTrue((np.array([1, 2, 3]) == d).all())

    def test_basic_write_context(self):
        handler = HDFFileHandler(fp1, 'r+')
        with handler as f:
            f.attrs['a'] = 'b'
            f['new_data'] = np.array([4, 5, 6])

        with h5py.File(fp1, 'r') as f:
            v = f.attrs['a']
            d = f['new_data'][:]

        self.assertEqual('b', v)
        self.assertTrue((np.array([4, 5, 6]) == d).all())

    def test_multithread_read_write(self):
        thread_pool = ThreadPoolExecutor(max_workers=10)

        def read_write_read(v, start_delay):
            handler = HDFFileHandler(fp1, 'r+')
            time.sleep(start_delay)
            with handler as f:
                f.attrs['test'] = v
                time.sleep(1)
                read = f.attrs['test']
            return read

        vs = np.linspace(0, 10, 10)
        delays = np.linspace(0, 1, 10)
        results = thread_pool.map(read_write_read, vs, delays)
        thread_pool.shutdown()
        for r, v in zip(results, vs):
            self.assertEqual(v, r)

    def test_multiprocess_read_write(self):
        process_pool = ProcessPoolExecutor(max_workers=5)

        vs = np.linspace(0, 10, 5)
        delays = np.linspace(0, 1, 5)
        results = process_pool.map(_test_read_write_read, vs, delays)
        process_pool.shutdown()
        for r, v in zip(results, vs):
            self.assertEqual(v, r)

    def test_nested_read_context(self):
        handler = HDFFileHandler(fp1, 'r')

        with handler as f:
            v = f.attrs['a']
            with handler as f2:
                d = f2['data'][:]

        self.assertEqual('a', v)
        self.assertTrue((np.array([1, 2, 3]) == d).all())
