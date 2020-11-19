from unittest import TestCase
import h5py
from tests import helpers
from src.DatObject.Attributes import Data
from src.HDF_Util import with_hdf_read

output_dir = 'Outputs/Data/'


class Testing_Data(Data.Data):
    """Override the normal init behaviour so it doesn't fail before reaching tests"""

    @with_hdf_read
    def check_init(self):
        group = self.hdf.get(self.group_name, None)
        if group is None:
            self._create_group(self.group_name)
            group = self.hdf.get(self.group_name)
        if group.attrs.get('initialized', False) is False:
            # self._initialize()  # This will run everything otherwise
            pass


class TestData(TestCase):
    helpers.clear_outputs(output_dir)
    dat = helpers.init_testing_dat(9111, output_directory=output_dir)
    D: Data.Data = Testing_Data(dat)


    def tearDown(self):
        """Runs AFTER every test"""
        with self.assertRaises(ValueError):
            filename = self.dat.hdf.hdf.filename  # Checking hdf is actually closed

    def test__set_exp_config_data_descriptors(self):
        self.D._set_exp_config_DataDescriptors()
        with h5py.File(self.D.hdf.hdf_path, 'r') as f:
            print(f.keys())
            g = f.get(self.D.group_name)#.get('Descriptors')
            print(g.keys())


    def test_keys(self):
        self.fail()

    def test_data_keys(self):
        self.fail()

    def test__get_data_keys(self):
        self.fail()

    def test_data_descriptors(self):
        self.fail()

    def test__get_all_descriptors(self):
        self.fail()

    def test_get_data_descriptor(self):
        self.fail()

    def test__get_default_descriptor_for_data(self):
        self.fail()

    def test__get_data_path_from_hdf(self):
        self.fail()

    def test_set_data_descriptor(self):
        self.fail()

    def test_get_data(self):
        self.fail()

    def test_get_orig_data(self):
        self.fail()

    def test_set_data(self):
        self.fail()

    def test_clear_caches(self):
        self.fail()

    def test__initialize_minimum(self):
        self.fail()
