from unittest import TestCase
from src.DatObject.DatHDF import DatHDFBuilder
from tests.helpers import get_testing_Exp2HDF
import os
import h5py
import numpy as np
import shutil
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
        from src.DatObject.DatHDF import DatHDF
        builder.init_DatHDF()
        self.assertIsInstance(builder.dat, DatHDF)

    def test_f_init_exp_config(self):
        from src.DataStandardize.ExpConfig import ExpConfigGroupDatAttribute
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
        from src.DatObject.DatHDF import DatHDF
        dat = builder.build_dat()
        assert isinstance(dat, DatHDF)
