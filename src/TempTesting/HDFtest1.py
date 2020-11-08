import h5py
import numpy as np
import src.CoreUtil as CU
import logging
import os
import time
import threading

logger = logging.getLogger(__name__)

logger.handlers = []  # Probably a bad thing to be doing...
handler = logging.StreamHandler()
formatter = logging.Formatter(f'%(threadName)s %(funcName)s %(lineno)d %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def make_test_file(array, i):
    with h5py.File(f'test{i}.h5', mode='w') as f:
        f['ds1'] = array


def make_test_files_concurrent(mode):
    array = np.linspace(0, 10, 10000000, dtype=np.float32)
    CU.run_concurrent([make_test_file]*5, [(array, i) for i in range(5)], which=mode)


def open_read(path):
    i = 0
    # print(os.path.abspath(f'/{path}'))
    while i < 100:
        i += 1
        with h5py.File(path, 'r+') as f:
            d = f['ds1'][:]
            if 'ds2' in f:
                del f['ds2']
            f['ds2'] = d
            if i % 100 == 0:
                logger.info(f'read_{i} done')


def open_wait(path):
    i = 0
    while i < 5:
        i += 1
        with h5py.File(path, 'r') as f:
            d = f['ds1'][:]
            time.sleep(1)
            logger.info(f'{i} -- opened and waited')


def open_wait2(path):
    i = 0
    while i < 5:
        i += 1
        # f = h5py.File(path, 'r+')
        with h5py.File(path, 'r+') as f:
            d = f['ds1'][:]
            if 'ds2' in f:
                prev_id = f['ds2'].attrs.get('Thread_id', None)
                del f['ds2']
            else:
                prev_id = None
            f['ds2'] = d
            id = threading.current_thread().ident
            f['ds2'].attrs['Thread_id'] = id

            # time.sleep(1)
            logger.info(f'{i} -- prev = {prev_id},  wrote id={id} and waited')
        # f.close()


if __name__ == '__main__':
    # t1 = time.time()
    # make_test_files_concurrent(mode='multithread')
    # print(time.time()-t1)
    # make_test_files_concurrent(mode='multiprocess')
    # print(time.time()-t1)
    # open_read('test1.h5')
    # print(time.time()-t1)
    # CU.run_concurrent([open_wait2]*10, func_args=[['test1.h5']]*10, which='multithread')
    # print(time.time()-t1)
    pass

threading.Thread(target=open_wait2, args=(('test0.h5')))
