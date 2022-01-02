from unittest import TestCase
from dat_analysis.dat_object.dat_hdf import DatHDF
from dat_analysis.dat_object.make_dat import get_dat, get_dats, DatHandler
from tests.helpers import get_testing_Exp2HDF
from dat_analysis.data_standardize.exp_specific.Feb21 import Feb21Exp2HDF
import concurrent.futures
import os
import h5py
import numpy as np
import shutil
import time
from tests import helpers

dat_dir = os.path.abspath('fixtures/dats/2021Feb')

# Where to put outputs (i.e. DatHDFs)
output_dir = os.path.abspath('Outputs/test_multithread_access')

Testing_Exp2HDF = get_testing_Exp2HDF(dat_dir, output_dir, base_class=Feb21Exp2HDF)
# SetUp before tests
helpers.clear_outputs(output_dir)
hdf_folder_path = os.path.join(output_dir, 'Dat_HDFs')


def read(datnum: DatHDF):
    dat = get_dat(datnum, exp2hdf=Testing_Exp2HDF)
    val = dat._threaded_read_test()
    return val

def write(datnum: DatHDF, value):
    dat = get_dat(datnum, exp2hdf=Testing_Exp2HDF)
    val = dat._threaded_write_test(value)
    return val


def mutithread_read(datnums):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(datnums) + 10) as executor:
        same_dat_results = [executor.submit(read, datnums[0]) for i in range(10)]
        diff_dat_results = [executor.submit(read, num) for num in datnums]

    return same_dat_results, diff_dat_results


class TestMultiAccess(TestCase):

    def setUp(self):
        self.dats = get_dats([717, 719, 720, 723, 724, 725], exp2hdf=Testing_Exp2HDF, overwrite=True)

    def tearDown(self) -> None:
        DatHandler().clear_dats()

    def set_test_attrs(self, dats, values):
        for dat, value in zip(dats, values):
            with h5py.File(dat.hdf.hdf_path, 'r+') as f:
                f.attrs['threading_test_var'] = value

    def test_threaded_read(self):
        """Check multiple read threads can run at the same time"""
        dats = self.dats
        values = [dat.datnum for dat in dats]
        self.set_test_attrs(dats, values)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.dats)+10) as executor:
            same_dat_results = [executor.submit(read, dats[0].datnum) for i in range(10)]
            diff_dat_results = [executor.submit(read, dat.datnum) for dat in dats]

        same_dat_results = [r.result() for r in same_dat_results]
        diff_dat_results = [r.result() for r in diff_dat_results]

        self.assertEqual(same_dat_results, [dats[0].datnum]*10)
        self.assertEqual(diff_dat_results, [dat.datnum for dat in dats])

    def test_threaded_write(self):
        """Check multiple threads trying to write at same time don't clash"""
        dats = self.dats
        values = ['not set' for dat in dats]
        self.set_test_attrs(dats, values)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            same_dat_writes = [executor.submit(write, dats[0].datnum, i) for i in range(10)]

        value = read(dats[0].datnum)
        self.assertTrue(value in [r.result() for r in same_dat_writes])  # Check that the final value was one of the writes at least

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.dats)) as executor:
            diff_dat_writes = executor.map(lambda args: write(*args), [(dat.datnum, dat.datnum) for dat in dats])

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.dats)) as executor:
            diff_dat_reads = executor.map(read, [dat.datnum for dat in dats])

        diff_dat_writes = [r for r in diff_dat_writes]
        diff_dat_reads = [r for r in diff_dat_reads]

        self.assertEqual(diff_dat_reads, diff_dat_writes)

    def test_multiprocess_read(self):
        """Check multiple read threads can run at the same time"""
        dats = self.dats
        values = [dat.datnum for dat in dats]
        self.set_test_attrs(dats, values)

        with concurrent.futures.ProcessPoolExecutor(max_workers=len(self.dats)+10) as executor:
            same_dat_results = [executor.submit(read, dats[0].datnum) for i in range(10)]
            diff_dat_results = [executor.submit(read, dat.datnum) for dat in dats]

        same_dat_results = [r.result() for r in same_dat_results]
        diff_dat_results = [r.result() for r in diff_dat_results]

        self.assertEqual(same_dat_results, [dats[0].datnum]*10)
        self.assertEqual(diff_dat_results, [dat.datnum for dat in dats])

    def test_multiprocess_write(self):
        """Check multiple threads trying to write at same time don't clash"""
        dats = self.dats
        values = ['not set' for dat in dats]
        self.set_test_attrs(dats, values)

        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            same_dat_writes = [executor.submit(write, dats[0].datnum, i) for i in range(10)]

        value = read(dats[0].datnum)
        self.assertTrue(value in [r.result() for r in same_dat_writes])  # Check that the final value was one of the writes at least

        with concurrent.futures.ProcessPoolExecutor(max_workers=len(self.dats)) as executor:
            diff_dat_writes = executor.map(lambda args: write(*args), [(dat.datnum, dat.datnum) for dat in dats])

        with concurrent.futures.ProcessPoolExecutor(max_workers=len(self.dats)) as executor:
            diff_dat_reads = executor.map(read, [dat.datnum for dat in dats])

        diff_dat_writes = [r for r in diff_dat_writes]
        diff_dat_reads = [r for r in diff_dat_reads]

        self.assertEqual(diff_dat_reads, diff_dat_writes)

    def test_hdf_write_inside_read(self):
        dat = self.dats[0]
        before, after = dat._write_inside_read_test()
        print(before, after)
        self.assertEqual(after, before + 1)

    def test_hdf_read_inside_write(self):
        dat = self.dats[0]
        before, after = dat._read_inside_write_test()
        print(before, after)
        self.assertEqual(after, before + 1)

    def test_multiprocess_multithread_read(self):
        dats = self.dats
        values = [dat.datnum for dat in dats]
        self.set_test_attrs(dats, values)
        datnums = [dat.datnum for dat in dats]

        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            results = [executor.submit(mutithread_read, datnums) for i in range(5)]

        for r in results:
            result = r.result()
            same_nums, diff_nums = result
            self.assertEqual(same_nums, [datnums[0]]*10)
            self.assertEqual(diff_nums, datnums)


