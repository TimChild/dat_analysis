from unittest import TestCase
import numpy as np
from src.HDF_Util import ThreadID, ThreadQueue
import threading
from concurrent.futures import ThreadPoolExecutor
from string import ascii_lowercase
import itertools


def iter_all_strings():
    """Generate incrementing strings"""
    for size in itertools.count(1):
        for s in itertools.product(ascii_lowercase, repeat=size):
            yield "".join(s)


class TestThreadID(TestCase):
    def test_init(self):
        """Test that inits can't mess each other up"""
        # target_modes = [s for s in itertools.islice(iter_all_strings(), 200)]  # Make 200 incrementing strings
        target_modes = [str(i) for i in range(1000)]
        with ThreadPoolExecutor(max_workers=100) as e:
            res = e.map(ThreadID, target_modes)
        res = [r for r in res]
        print([r.target_mode for r in res])
        self.assertEqual(target_modes, [r.target_mode for r in res])


class TestThreadQueue(TestCase):
    def test_init(self):
        """with locking it should pass"""
        lock = threading.Lock()
        inst = None

        class Temp:
            def __init__(self, queue):
                self.queue = queue

        def init_temps():
            with lock:
                nonlocal inst
                arr = np.ones((100, 5000))
                if inst is None:
                    arr = arr**2  # Slow things down here
                    inst = ThreadQueue()
                return Temp(inst)

        with ThreadPoolExecutor(100) as e:
            res = [e.submit(init_temps) for _ in range(1000)]
        res = [r.result() for r in res]
        res[0].queue.put(ThreadID('test'))
        matching = np.sum([True if len(r.queue.queue) == 1 else False for r in res])
        self.assertTrue(1000 == matching)

    def test_init_fail_without_lock(self):
        """Without locking the initial init it will fail"""
        inst = None

        class Temp:
            def __init__(self, queue):
                self.queue = queue

        def init_temps():
            nonlocal inst
            arr = np.ones((100, 5000))
            if inst is None:
                arr = arr**2  # Slow things down here
                inst = ThreadQueue()
            return Temp(inst)

        with ThreadPoolExecutor(100) as e:
            res = [e.submit(init_temps) for _ in range(1000)]
        res = [r.result() for r in res]
        res[0].queue.put(ThreadID('test'))
        matching = np.sum([True if len(r.queue.queue) == 1 else False for r in res])
        self.assertFalse(1000 == matching)