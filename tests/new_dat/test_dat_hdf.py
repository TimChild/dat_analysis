from unittest import TestCase
from dat_analysis.new_dat.dat_hdf import DatHDF
import os

test_hdf_filepath = 'test_hdf.h5'


def rebuild_test_hdf():
    """Rebuilds the test hdf file"""
    import h5py
    with h5py.File(test_hdf_filepath, 'w') as f:
        f.attrs['test_top_attr'] = 'test_top_attr'
        f.require_group('Logs')
        f.require_group('Data')


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

        with self.assertRaises(OSError) as context:
            with self.dat.hdf_read as f:
                f.attrs['new_attr'] = 'new_value'

    def test_hdf_write(self):
        with self.dat.hdf_write as f:
            f.attrs['new_attr'] = 'new_value'

        with self.dat.hdf_read as f:
            new_attr = f.attrs['new_attr']

        self.assertEqual('new_value', new_attr)
