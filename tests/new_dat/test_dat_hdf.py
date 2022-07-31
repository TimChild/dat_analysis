# from unittest import TestCase
# from dat_analysis.new_dat.dat_hdf import DatHDF, get_dat, get_dat_from_exp_filepath
# from dat_analysis.new_dat.new_dat_util import get_local_config
# import os
# import h5py
# from .helper import setup_test_config
#
# # Change to a testing config.toml file
# setup_test_config()
#
#
# def rebuild_test_hdf():
#     """Rebuilds the test hdf file"""
#     config = get_local_config()
#     test_filename = 'dat1.h5'
#     with h5py.File(os.path.join(config['loading']['path_to_save_directory'], test_filename), 'w') as f:
from unittest import TestCase
from dat_analysis.new_dat.dat_hdf import DatHDF, get_dat, get_dat_from_exp_filepath
import dat_analysis.new_dat.new_dat_util as ndu
import os
from .helper import setup_test_config

# Change to a testing config.toml file
setup_test_config()

test_hdf_filepath = os.path.normpath(os.path.join(__file__, '../../Outputs/new_dat/dat1.h5'))
experiment_data_file = os.path.normpath(os.path.join(__file__, "../../fixtures/test_measurement-data/test-pc/test-user/test-experiment/dat1.h5"))


def rebuild_test_hdf():
    """Rebuilds the test hdf file"""
    import h5py
    os.makedirs(os.path.dirname(test_hdf_filepath), exist_ok=True)
    with h5py.File(test_hdf_filepath, 'w') as f:
        f.attrs['experiment_data_path'] = ''
        f.require_group('Logs')
        f.require_group('Data')

        f.attrs['test_top_attr'] = 'test_top_attr'


class TestDatHDF(TestCase):

    def setUp(self) -> None:
        rebuild_test_hdf()
        self.dat = DatHDF(test_hdf_filepath)

    def tearDown(self) -> None:
        os.remove(test_hdf_filepath)

    def test_hdf_read(self):
        """Open HDF and read a value (does not test multi threading/processing)"""
        with self.dat.hdf_read as f:
            top_attr = f.attrs['test_top_attr']
            self.assertEqual('test_top_attr', top_attr)

        with self.assertRaises(RuntimeError):  # Check can't write to 'r' file  (h5py should raise RuntimeError: Unable to create attribute (no write intent on file)
            with self.dat.hdf_read as f:
                f.attrs['new_attr'] = 'new_value'

    def test_hdf_write(self):
        with self.dat.hdf_write as f:
            f.attrs['new_attr'] = 'new_value'

        with self.dat.hdf_read as f:
            new_attr = f.attrs['new_attr']

        self.assertEqual('new_value', new_attr)

    def test_hdf_context(self):
        """Using the DatHDF as a context manager for accessing HDF file"""
        self.dat.mode = 'r'
        with self.dat as f:
            attr = f.attrs['test_top_attr']

        self.assertEqual('test_top_attr', attr)

        with self.assertRaises(RuntimeError):  # Check can't write to 'r' file  (h5py should raise RuntimeError: Unable to create attribute (no write intent on file)
            with self.dat as f:
                f.attrs['new_attr'] = 'new'

        self.dat.mode = 'r+'
        with self.dat as f:
            f.attrs['another_new_attr'] = 'new'
            attr = f.attrs['another_new_attr']
        self.assertEqual('new', attr)


class TestGeneral(TestCase):
    def setUp(self) -> None:
        rebuild_test_hdf()

    def test_get_dat(self):
        dat = get_dat(datnum=1, host_name='test-pc', user_name='test-user', experiment_name='test-experiment')
        self.assertIsInstance(dat, DatHDF)

    def test_get_dat_from_exp_filepath(self):
        dat = get_dat_from_exp_filepath(experiment_data_path=experiment_data_file)
        self.assertIsInstance(dat, DatHDF)


