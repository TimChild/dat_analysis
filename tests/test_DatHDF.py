from unittest import TestCase
from dat_analysis.dat_object.dat_hdf import DatHDFBuilder, DatHDF
from tests.helpers import get_testing_Exp2HDF
from typing import List
import os
import h5py
import numpy as np
import shutil
import time
from tests import helpers

dat_dir = os.path.abspath('fixtures/dats/2020Sep')
"""
Contents of dat_dir relevant in this file:
    Dat9111: Square entropy with dS ~Ln2

"""

# Where to put outputs (i.e. DatHDFs)
output_dir = os.path.abspath('Outputs/test_DatHDFBuilder')
print(os.path.abspath('unit'))

Testing_Exp2HDF = get_testing_Exp2HDF(dat_dir, output_dir)

# SetUp before tests
helpers.clear_outputs(output_dir)
exp2hdf = Testing_Exp2HDF(9111, 'base')
builder = DatHDFBuilder(exp2hdf, 'min')
hdf_folder_path = os.path.join(output_dir, 'Dat_HDFs')
dat_hdf_path = os.path.join(hdf_folder_path, 'dat9111.h5')  # if datname=='base' it's not in filepath


class TestDatHDFBuilder(TestCase):

    def _del_hdf_contents(self):
        if os.path.exists(hdf_folder_path):
            for root, dirs, files in os.walk(hdf_folder_path):
                for f in files:
                    os.remove(os.path.join(hdf_folder_path, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(hdf_folder_path, d))

    def setUp(self):
        """Runs before every test"""
        pass

    def test_a0_create_hdf_fails_no_path(self):
        if os.path.isdir(hdf_folder_path):
            shutil.rmtree(hdf_folder_path)
        with self.assertRaises(NotADirectoryError):
            builder.create_hdf()
        os.makedirs(hdf_folder_path, exist_ok=True)

    def test_a1_create_hdf(self):
        os.makedirs(hdf_folder_path, exist_ok=True)
        self._del_hdf_contents()
        self.assertFalse(os.path.isfile(dat_hdf_path))
        builder.create_hdf()
        self.assertTrue(os.path.isfile(dat_hdf_path))

    def test_b_create_hdf_overwrite_fail(self):
        self.assertTrue(os.path.isfile(dat_hdf_path))  # This needs to be True to do the test
        with self.assertRaises(FileExistsError):
            builder.create_hdf()

    def test_c_hdf_openable(self):
        with h5py.File(dat_hdf_path, 'r+') as f:
            pass
        self.assertTrue(True)

    def test_d_copy_exp_data(self):
        builder.copy_exp_data()
        with h5py.File(dat_hdf_path, 'r') as f:
            copied_data = f.get('Experiment Copy')
            data = copied_data.get('cscurrent_2d')[0, :]
        self.assertIsInstance(data, np.ndarray)

    def test_d_init_dat_hdf(self):
        from dat_analysis.dat_object.dat_hdf import DatHDF
        builder.init_DatHDF()
        self.assertIsInstance(builder.dat, DatHDF)

    def test_f_init_exp_config(self):
        from dat_analysis.data_standardize.exp_config import ExpConfigGroupDatAttribute
        builder.init_ExpConfig()
        self.assertIsInstance(builder.dat.ExpConfig, ExpConfigGroupDatAttribute)

    def test_g_other_inits(self):
        builder.other_inits()
        self.assertTrue(True)  # Just check this runs

    def test_h_init_base_attrs(self):
        builder.init_base_attrs()
        with h5py.File(dat_hdf_path, 'r') as f:
            attrs = f.attrs
            exp_val = {
                9111: attrs.get('datnum'),
                'base': attrs.get('datname'),
            }
        for k, v in exp_val.items():
            self.assertEqual(k, v)

    def test_i__get_base_attrs(self):
        attr_dict = builder._get_base_attrs()
        self.assertIsInstance(attr_dict, dict)

    def test_j_build_dat(self):
        self._del_hdf_contents()
        from dat_analysis.dat_object.dat_hdf import DatHDF
        dat = builder.build_dat()
        assert isinstance(dat, DatHDF)


class TestThreading(TestCase):
    from dat_analysis.dat_object.make_dat import DatHandler, get_dat, get_dats
    from dat_analysis.data_standardize.exp_specific.Feb21 import Feb21Exp2HDF
    from concurrent.futures import ThreadPoolExecutor

    dat_dir = os.path.abspath('fixtures/dats/2021Feb')

    # Where to put outputs (i.e. DatHDFs)
    Testing_Exp2HDF = get_testing_Exp2HDF(dat_dir, output_dir, base_class=Feb21Exp2HDF)

    pool = ThreadPoolExecutor(max_workers=5)
    different_dats = get_dats([717, 719, 720, 723, 724, 725], exp2hdf=Testing_Exp2HDF)
    single_dat = different_dats[0]
    same_dats = [single_dat] * 10

    def test_threaded_manipulate_test(self):
        """Test that running multiple threads through a method which changes an instance attribute works with
        thread locks"""
        def threaded_manipulate_test(dat: DatHDF):
            eq = dat._threaded_manipulate_test()
            return eq

        t1 = time.time()
        rets = list(self.pool.map(threaded_manipulate_test, self.different_dats))
        print(f'Time elapsed: {time.time()-t1:.2f}s, Returns = {rets}')
        self.assertTrue(all(rets))

        t1 = time.time()
        rets = list(self.pool.map(threaded_manipulate_test, self.same_dats))
        print(f'Time elapsed: {time.time()-t1:.2f}s, Returns = {rets}')
        self.assertTrue(all(rets))

    def test_threaded_reentrant_test(self):
        """Test that the reentrant lock allows a recursive method call to work properly"""
        t1 = time.time()
        ret = self.single_dat._threaded_reentrant_test(i=0)
        print(f'Time elapsed: {time.time()-t1:.2f}s, Returns = {ret}')
        self.assertEqual(3, ret)

        def reentrant_test(dat: DatHDF, i):
            return dat._threaded_reentrant_test(i=i)

        t1 = time.time()
        rets = list(self.pool.map(reentrant_test, self.different_dats, [0, 7, 1, 5, 1, 7]))
        print(f'Time elapsed: {time.time()-t1:.2f}s, Returns = {rets}')
        self.assertEqual([3, 7, 3, 5, 3, 7], rets)

        t1 = time.time()
        rets = list(self.pool.map(reentrant_test, self.same_dats, [0, 7, 1, 5, 1, 7]))
        print(f'Time elapsed: {time.time()-t1:.2f}s, Returns = {rets}')
        self.assertEqual([3, 7, 3, 5, 3, 7], rets)

    # def test_threaded_read_test(self):
    #     """Check that multiple reads on the same/different HDFs can be carried out simultaneously (reading same attr)"""
    #     def setup_variables(dats: List[DatHDF], values=None):
    #         """Single threaded writing of variable to HDF as initialization"""
    #         if values is None:
    #             values = list(range(len(dats)))
    #         for dat, value in zip(dats, values):
    #             with h5py.File(dat.hdf.hdf_path, 'r+') as f:
    #                 f.attrs['threading_test_var'] = value
    #
    #     def threaded_read(dat: DatHDF):
    #         return dat._threaded_read_test()
    #
    #
    #     t1 = time.time()
    #     dat = self.different_dats[0]
    #     print(dat.datnum)
    #     setup_variables([dat], values=None)
    #     rets = dat._threaded_read_test()
    #     print(f'Time elapsed: {time.time()-t1:.2f}s, Returns = {rets}')
    #     self.assertEqual(0, rets)
    #
    #     t1 = time.time()
    #     diff_dats = self.different_dats
    #     setup_variables(diff_dats, values=None)
    #     rets = list(self.pool.map(threaded_read, diff_dats))
    #     print(f'Time elapsed: {time.time()-t1:.2f}s, Returns = {rets}')
    #     self.assertEqual(list(range(len(diff_dats))), rets)
    #
    #     t1 = time.time()
    #     same_dats = self.same_dats
    #     setup_variables([same_dats[0]], values=[10])
    #     rets = list(self.pool.map(threaded_read, same_dats))
    #     print(f'Time elapsed: {time.time()-t1:.2f}s, Returns = {rets}')
    #     self.assertEqual([10]*len(same_dats), rets)
    #
    # def test_threaded_write_test(self):
    #     """Check that writing to to same/different HDFs is handled properly (only one write should take place at a time)"""
    #     def setup_variables(dats: List[DatHDF]):
    #         """Set variable to 'not set' initially with a single thread"""
    #         for dat in dats:
    #             with h5py.File(dat.hdf.hdf_path, 'r+') as f:
    #                 f.attrs['threading_test_var'] = 'not set'
    #
    #     def write_only(dat: DatHDF, value):
    #         dat._threaded_write_test(value)
    #         return value
    #
    #     def read_only(dat: DatHDF):
    #         return dat._threaded_read_test()
    #
    #     diff_dats = self.different_dats
    #     setup_variables(diff_dats)
    #     t1 = time.time()
    #     writes = list(self.pool.map(write_only, diff_dats, range(len(diff_dats))))
    #     reads = list(self.pool.map(read_only, diff_dats))
    #     print(f'Time elapsed: {time.time()-t1:.2f}s, Returns = {reads}')
    #     self.assertEqual(writes, reads)
    #
    #     same_dats = self.same_dats
    #     setup_variables([same_dats[0]])
    #     t1 = time.time()
    #     writes = list(self.pool.map(write_only, same_dats, range(len(same_dats))))
    #     reads = list(self.pool.map(read_only, same_dats))
    #     print(f'Time elapsed: {time.time()-t1:.2f}s, Returns = {reads}')
    #     self.assertEqual([len(same_dats)-1]*len(same_dats), reads)
    #
    # def test_hdf_write_inside_read(self):
    #     dat = self.different_dats[0]
    #     before, after = dat._write_inside_read_test()
    #     print(before, after)
    #     self.assertEqual(after, before+1)
    #
    # def test_hdf_read_inside_write(self):
    #     dat = self.different_dats[0]
    #     before, after = dat._read_inside_write_test()
    #     print(before, after)
    #     self.assertEqual(after, before+1)
    #
    # def test_hdf_write_read_same_time(self):
    #     import random
    #     import threading
    #     NUM = 100
    #
    #     lock = threading.Lock()
    #
    #     def initial_setup(path):
    #         with h5py.File(path, 'w') as f:
    #             for i in range(NUM):
    #                 f.attrs[str(i)] = i
    #
    #     def read(path):
    #         with h5py.File(path, 'r') as f:
    #             ret = list()
    #             for i in range(NUM):
    #                 ret.append(f.attrs.get(str(i)))
    #         return ret
    #
    #     def write(path):
    #         with lock:
    #             f = h5py.File(path, 'r+')
    #             print('writing')
    #             time.sleep(random.random()*0.1)
    #             v1 = f.attrs['1']
    #             for i in range(NUM):
    #                 v = f.attrs.get(str(i), 0)
    #                 time.sleep(random.random() * 0.001)
    #                 f.attrs[str(i)] = v*2
    #             print('done writing')
    #             f.close()
    #             return v1
    #
    #
    #     path = 'temp.h5'
    #     initial_setup(path)
    #     rs = list(self.pool.map(read, [path]*5))
    #     print(f'0s = {[r[0] for r in rs]}\n'
    #           f'5s = {[r[5] for r in rs]}\n'
    #           f'-5s = {[r[-5] for r in rs]}\n'
    #           f'-1s = {[r[-1] for r in rs]}\n')
    #     ws = list(self.pool.map(write, [path]*5))
    #     print(ws)
    #
    #     r = read(path)
    #     print(f'{r[:10], r[-10:]}')
    #
    #     # print(w, r[:5], r[-5:])
    #
    #


