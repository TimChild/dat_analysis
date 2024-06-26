from unittest import TestCase
import os
import time
import h5py
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dat_analysis.hdf_file_handler import HDFFileHandler, FlexibleFile, _wait_until_free
from dat_analysis.useful_functions import set_default_logging
set_default_logging()

# Not permanent, free to be deleted in tests etc
output_path = os.path.normpath(os.path.join(__file__, '../Outputs/hdf_file_manager/'))
os.makedirs(output_path, exist_ok=True)

fp1 = os.path.join(output_path, 'f1.h5')


def _test_write_then_read(v, start_delay):
    time.sleep(start_delay)
    with HDFFileHandler(fp1, 'r+') as f:
        f.attrs['test'] = v
        time.sleep(1)
        read = f.attrs['test']
    return read


def read_only(delay):
    with HDFFileHandler(fp1, 'r') as f:
        time.sleep(delay)
        read = f.attrs['a']
    return read


def write_only(delay):
    with HDFFileHandler(fp1, 'r+') as f:
        time.sleep(delay)
        f.attrs['test_write'] = 1
    return 1


class TestHDFFileHandler(TestCase):

    def setUp(self) -> None:
        print('setting up')
        with h5py.File(fp1, 'w') as f:
            f.attrs['a'] = 'a'
            f['data'] = np.array([1, 2, 3])
            f.require_group('group_1')
        print('done setting up')

    def tearDown(self) -> None:
        print('tearing down')
        time.sleep(0.2)  # Some time for threads to end
        if os.path.exists(fp1):
            os.remove(fp1)

    def test_basic_read_context(self):
        handler = HDFFileHandler(fp1, 'r')
        with handler as f:
            v = f.attrs['a']
            d = f['data'][:]

        self.assertEqual('a', v)
        self.assertTrue((np.array([1, 2, 3]) == d).all())
        self.assertFalse(bool(f))

    def test_basic_write_context(self):
        print('starting test basic write')
        handler = HDFFileHandler(fp1, 'r+')
        with handler as f:
            f.attrs['a'] = 'b'
            f['new_data'] = np.array([4, 5, 6])

        with h5py.File(fp1, 'r') as f:
            v = f.attrs['a']
            d = f['new_data'][:]

        self.assertEqual('b', v)
        self.assertTrue((np.array([4, 5, 6]) == d).all())
        self.assertFalse(bool(f))

    def test_basic_read_then_write(self):
        print('starting test basic read then write')
        with HDFFileHandler(fp1, 'r') as f:
            v = f.attrs['a']
            d = f['data'][:]

        self.assertEqual('a', v)
        self.assertTrue((np.array([1, 2, 3]) == d).all())

        with HDFFileHandler(fp1, 'r+') as f:
            f.attrs['a'] = 'b'
            f['new_data'] = np.array([4, 5, 6])

        with h5py.File(fp1, 'r') as f:
            v = f.attrs['a']
            d = f['new_data'][:]

        self.assertEqual('b', v)
        self.assertTrue((np.array([4, 5, 6]) == d).all())
        self.assertFalse(bool(f))

    def test_multithread_read_write(self):
        """Check that a write followed by a delayed read still gets the same value (i.e. no other
        writing threads can change the value in between)"""
        print('starting multithread read then write')
        num = 3
        thread_pool = ThreadPoolExecutor(max_workers=num)

        vs = np.linspace(0, 10, num)
        delays = np.linspace(0, 1, num)
        results = thread_pool.map(_test_write_then_read, vs, delays)
        thread_pool.shutdown(wait=True)
        for r, v in zip(results, vs):
            self.assertEqual(v, r)

    def test_multiprocess_read_write(self):
        print('starting multiprocess read then write')
        process_pool = ProcessPoolExecutor(max_workers=5)

        vs = np.linspace(0, 10, 5)
        delays = np.linspace(0, 1, 5)
        results = process_pool.map(_test_write_then_read, vs, delays)
        process_pool.shutdown(wait=True)
        for r, v in zip(results, vs):
            self.assertEqual(v, r)

    def test_write_waits_for_read_threading(self):
        """Test that a pending write waits for read threads to finish"""
        print('starting write wait for read threading')
        thread_pool = ThreadPoolExecutor(max_workers=2)

        individual_delay = 1

        start_time = time.time()
        future_read = thread_pool.submit(read_only, individual_delay)
        time.sleep(0.1)  # some time for read to definitely start
        future_write = thread_pool.submit(write_only, individual_delay)
        thread_pool.shutdown(wait=True)
        self.assertEqual('a', future_read.result())
        self.assertEqual(1, future_write.result())
        self.assertGreater(time.time() - start_time, individual_delay*2)  # Shouldn't be able to finish faster than this
        self.assertLess(time.time() - start_time, individual_delay*5)  # Shouldn't take a really long time

    def test_multiple_read_threading(self):
        """Test that multiple reads can take place at the same time"""
        print('starting multiple read threading')
        num = 10
        thread_pool = ThreadPoolExecutor(max_workers=num)
        single_delay = 1

        delays = np.ones(num)*single_delay
        start = time.time()
        results = thread_pool.map(read_only, delays)
        thread_pool.shutdown(wait=True)
        for r in results:
            self.assertEqual('a', r)
        self.assertLess(time.time() - start, 1.1*single_delay)
        print('done multiple read threading')

    def test_nested_read_context_new_each_time(self):
        """Should be able to create nested filemanagers for a single file and read from each"""
        print('starting nested read context')
        with HDFFileHandler(fp1, 'r') as f:
            v = f.attrs['a']
            with HDFFileHandler(fp1, 'r') as f2:
                d = f2['data'][:]
            v2 = f.attrs['a']

        self.assertEqual('a', v)
        self.assertEqual('a', v2)
        self.assertTrue((np.array([1, 2, 3]) == d).all())
        self.assertFalse(bool(f))
        self.assertFalse(bool(f2))

    def test_nested_read_context_repeated(self):
        """should raise error if same handler is used in nested way"""
        print('starting nested read context repeated')
        handler = HDFFileHandler(fp1, 'r')

        with self.assertRaises(RuntimeError):
            with handler as f:
                v = f.attrs['a']
                with handler as f2:
                    d = f2['data'][:]
                v2 = f.attrs['a']
        self.assertFalse(bool(f))

    def test_nested_read_write(self):
        """Test that a write context can be used inside a read context"""
        print('starting nested read write')
        with HDFFileHandler(fp1, 'r') as f:
            v = f.attrs['a']
            with HDFFileHandler(fp1, 'r+') as f2:
                f2.attrs['a'] = 'b'
            v2 = f.attrs['a']

        self.assertEqual('a', v)
        self.assertEqual('b', v2)
        self.assertFalse(bool(f))
        self.assertFalse(bool(f2))

    def test_nested_write_read(self):
        """Test that write intent is maintained inside a nested read"""
        print('starting nested write read')
        with HDFFileHandler(fp1, 'r+') as f:
            f.attrs['a'] = 'b'
            with HDFFileHandler(fp1, 'r') as f2:  # Note: Intended that f2 keeps write intent s.t. f stays write
                v = f2.attrs['a']
                f.attrs['b'] = v
            v2 = f.attrs['a']
            v3 = f.attrs['b']

        self.assertEqual('b', v)
        self.assertEqual('b', v2)
        self.assertEqual('b', v3)
        self.assertFalse(bool(f))
        self.assertFalse(bool(f2))

    def test_switch_to_write_in_read(self):
        """Check that file can be switched to write mode
        such that the original read stays open after"""
        print('starting switch to write in read')
        with HDFFileHandler(fp1, 'r') as f:
            with HDFFileHandler(fp1, 'r+') as f2:
                f2.attrs['a'] = 'b'
            v = f.attrs['a']
        self.assertEqual('b', v)
        self.assertFalse(bool(f))
        self.assertFalse(bool(f2))

    def test_switch_to_read_in_write(self):
        """Check that file can be switched to read mode
        such that the original write stays open after"""
        print('starting switch to read in write')
        with HDFFileHandler(fp1, 'r+') as f:
            f.attrs['a'] = 'b'
            with HDFFileHandler(fp1, 'r') as f2:
                v = f2.attrs['a']
            f.attrs['a'] = ['c']
        self.assertEqual('b', v)
        self.assertFalse(bool(f))
        self.assertFalse(bool(f2))

    def test_multiple_read_then_write(self):
        """Check that multiple reads can be opened followed by a write in a single thread and it won't get locked"""
        print('staring multiple read then write')
        with HDFFileHandler(fp1, 'r') as f1:
            with HDFFileHandler(fp1, 'r') as f2:
                with HDFFileHandler(fp1, 'r+') as f3:
                    f3.attrs['b'] = 1
                f2v = f2.attrs['b']
            f1v = f1.attrs['b']
        self.assertEqual(1, f2v)
        self.assertEqual(1, f1v)
        self.assertFalse(bool(f1))
        self.assertFalse(bool(f2))
        self.assertFalse(bool(f3))

    def test_waits_for_file_to_close(self):
        """Check that it will wait for the file to be closed from somewhere else (e.g. simulating if open in HDFViewer)
        For read it should not wait
        for write it should wait
        """
        ####################
        # Having this run before was causing an error previously, so might as well leave this test in here
        self.test_switch_to_write_in_read()
        self.tearDown()
        self.setUp()
        ##################
        start = time.time()
        with h5py.File(fp1, 'r') as f1:
            print('File opened, about to open in R mode')
            with HDFFileHandler(fp1, 'r') as f2:  # This should not wait
                print('File opened in R mode')
                v = f2.attrs['a']
            self.assertEqual('a', v)

            print('About to open in R+ mode')
            with self.assertRaises(TimeoutError):
                with HDFFileHandler(fp1, 'r+', open_file_timeout=1) as f3:  # This should wait and return TimeoutError
                    print('File opened in R+ mode')
                    pass
            print('Done')
            for thread in threading.enumerate():
                print(thread.name, thread.ident)

        print(time.time()-start)
        self.assertLess(time.time() - start, 2)  # Should timeout in ~1s
        time.sleep(5)  # For debugging messages to come through

    def test_write_of_same_thread(self):
        """Check that a second write request from the same thread is not blocked by other write requests"""
        def padded_print(msg):
            time.sleep(0.3)
            print(msg)
            time.sleep(0.3)

        def first_writer(event: threading.Event):
            with HDFFileHandler(fp1, 'r+') as f:
                padded_print(f'First writer doing first write')
                f.attrs['first'] = 1
                event.wait()
                padded_print(f'First writer starting second write')
                with HDFFileHandler(fp1, 'r+') as f2:
                    f.attrs['second'] = 2
            padded_print(f'First writer done')

        def second_writer(event: threading.Event()):
            event.set()
            padded_print(f'Second writer starting first write')
            with HDFFileHandler(fp1, 'r+') as f:
                f.attrs['first'] = 1
            padded_print(f'Second writer done')

        start = time.time()
        e = threading.Event()
        t1 = threading.Thread(target=first_writer, args=(e,))
        t2 = threading.Thread(target=second_writer, args=(e,))
        t1.start()
        t2.start()
        t1.join(timeout=2)
        t2.join(timeout=2)
        self.assertLess(time.time()-start, 2)
        self.assertTrue(True)

    def test_EXPECTED_TO_FAIL_two_read_writes_blocking(self):
        """Two separate threads can lock each other by both starting in read and then both wanting to write --
        currently an unsolved problem

            *******************************************************************
        WARNING WARNING WARNING:
            I know that this test fails... I just don't know how/want to fix it yet

            this is just to remind me that this is a known issue... but it shouldn't come up much in real use.
            *******************************************************************

        """
        raise RuntimeError(f'NOTE: This test currently fails, so just raising an error early.')
        def padded_print(msg):
            time.sleep(0.3)
            print(msg)
            time.sleep(0.3)

        def first_thread(event1, event2):
            with HDFFileHandler(fp1, 'r') as f1:
                padded_print('first thread setting trigger for second thread')
                event1.set()
                padded_print('first thread waiting for trigger from second thread')
                event2.wait()
                with HDFFileHandler(fp1, 'r+') as f2:
                    f2.attrs['b'] = 1
                v = f1.attrs['b']
            padded_print('first thread done')
            return v

        def second_thread(event1, event2):
            padded_print('second thread waiting for trigger from first thread')
            event1.wait()
            with HDFFileHandler(fp1, 'r') as f1:
                padded_print('second thread setting trigger for first thread')
                event2.set()
                with HDFFileHandler(fp1, 'r+') as f2:
                    f2.attrs['b'] = 2
                v = f1.attrs['b']
            padded_print('second thread done')
            return v

        start = time.time()
        e1, e2 = threading.Event(), threading.Event()
        t1 = threading.Thread(target=first_thread, args=(e1, e2))
        t2 = threading.Thread(target=second_thread, args=(e1, e2))
        t1.start()
        t2.start()
        t1.join(timeout=2)
        t2.join(timeout=2)

        # Do some cleanup when this test fails (which it currently always does)
        if time.time() - start > 2:
            print('Test failing, just need to cleanup so file can be deleted')
            fh = HDFFileHandler(fp1, 'r')
            fq = fh._get_file_queue(fp1)
            fq.kill(fh.get_file(fp1))

        self.assertLess(time.time()-start, 2)
