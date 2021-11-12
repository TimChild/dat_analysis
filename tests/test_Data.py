from unittest import TestCase
import numpy as np
import src.hdf_util as HDU
import h5py
from tests import helpers
from src.dat_object.attributes import Data
from src.hdf_util import with_hdf_read
from src.dat_object.attributes.DatAttribute import DataDescriptor
from src.data_standardize.exp_config import DataInfo

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

    def setUp(self) -> None:
        """Runs BEFORE every test"""
        pass

    def tearDown(self):
        """Runs AFTER every test"""
        with self.assertRaises(ValueError):
            filename = self.dat.hdf.hdf.filename  # Checking hdf is actually closed

    def test__set_exp_config_data_descriptors(self):
        self.D._set_exp_config_DataDescriptors()
        with h5py.File(self.D.hdf.hdf_path, 'r') as f:
            g = f.get(self.D.group_name).get('Descriptors')
            expected_descriptor = DataDescriptor(data_path='/Experiment Copy/cscurrent_2d')
            descriptor = HDU.get_attr(g, 'i_sense', dataclass=DataDescriptor)
            self.assertEqual(descriptor, expected_descriptor)

    def test_keys(self):
        self.D.initialize_minimum()

        print(self.D.keys)
        self.assertEqual(set(), {'y', 'Experiment Copy/x_array', 'i_sense', 'Experiment Copy/fdAW_1', 'x', 'Experiment Copy/fdAW_0', 'x_array', 'Experiment Copy/cscurrent_2d', 'Experiment Copy/y_array'}
                         - set(self.D.keys))

    def test_data_keys(self):
        self.D.initialize_minimum()
        keys = self.D.data_keys
        self.assertEqual(set(), {'Experiment Copy/cscurrent_2d', 'Experiment Copy/fdAW_0', 'Experiment Copy/fdAW_1',
                           'Experiment Copy/x_array', 'Experiment Copy/y_array'} - set(keys))

    def test__get_data_keys(self):
        keys = self.D._get_data_keys()
        expected = ['Experiment Copy/cscurrent_2d',
                    'Experiment Copy/fdAW_0',
                    'Experiment Copy/fdAW_1',
                    'Experiment Copy/x_array',
                    'Experiment Copy/y_array']
        self.assertEqual(expected, keys)

    def test_data_descriptors(self):
        self.D.initialize_minimum()
        descriptors = self.D.data_descriptors
        expected_descriptor = DataDescriptor('/Experiment Copy/cscurrent_2d')
        expected_descriptor_hash = hash(expected_descriptor)
        self.assertTrue(expected_descriptor_hash in [hash(d) for d in descriptors.values()])

    def test__get_all_descriptors(self):
        self.D.initialize_minimum()
        descriptors = self.D._get_all_descriptors()
        expected_descriptor = DataDescriptor('/Experiment Copy/cscurrent_2d')
        # expected_descriptor_hash = hash(expected_descriptor)
        # print([expected_descriptor == d for d in descriptors.values()])
        # print([expected_descriptor is d for d in descriptors.values()])
        # print([hash(expected_descriptor) == hash(d) for d in descriptors.values()])
        # print(f'{expected_descriptor}\n{list(descriptors.values())[0]}')
        # print(expected_descriptor.__eq__(list(descriptors.values())[0]))
        # print(hash(expected_descriptor), hash(list(descriptors.values())[0]))
        # print(expected_descriptor.__class__, list(descriptors.values())[0].__class__)
        self.assertTrue(expected_descriptor in descriptors.values())

    def test_get_data_descriptor(self):
        self.D.initialize_minimum()
        d = self.D.get_data_descriptor('i_sense')
        expected = DataDescriptor('/Experiment Copy/cscurrent_2d')
        self.assertEqual(d, expected)

    def test__get_default_descriptor_for_data(self):
        # self.D.initialize_minimum()
        descriptors = [self.D._get_default_descriptor_for_data(name) for name in self.D.data_keys]
        expected_descriptor = DataDescriptor('/Experiment Copy/cscurrent_2d')
        expected_descriptor_hash = hash(expected_descriptor)
        self.assertTrue(expected_descriptor in descriptors)

    def test__get_data_path_from_hdf(self):
        path = self.D._get_data_path_from_hdf('cscurrent_2d')
        self.assertEqual(path, '/Experiment Copy/cscurrent_2d')

    def test_set_data_descriptor(self):
        self.D.initialize_minimum()
        descriptor = DataDescriptor('/Experiment Copy/cscurrent_2d', offset=10, multiply=20)
        self.D.set_data_descriptor(descriptor, name='test')
        d = self.D.get_data_descriptor('test', filled=False)
        self.assertEqual(descriptor, d)

    def test_DataDescriptor_multiply_offset(self):
        self.D.initialize_minimum()
        descriptor = DataDescriptor('/Experiment Copy/cscurrent_2d', offset=10, multiply=20)
        self.D.set_data_descriptor(descriptor, name='test')
        d = self.D.get_data_descriptor('test', filled=True)
        o_d = self.D.get_data_descriptor('i_sense', filled=True)
        self.assertTrue(np.allclose((o_d.data + 10) * 20, d.data, atol=1e-5))

    def test_set_data_descriptor_raises(self):
        with self.assertRaises(FileNotFoundError):
            descriptor = DataDescriptor('notafile')
            self.D.set_data_descriptor(descriptor, name='test2')

    def test_get_data(self):
        self.D.initialize_minimum()
        isense = self.D.get_data('i_sense')
        orig_x = self.D.get_data('x_array')
        x = self.D.get_data('x')
        self.assertTrue(np.all(orig_x == x))
        self.assertEqual(isense.shape, (50, 72816))

    def test_get_orig_data(self):
        self.D.initialize_minimum()
        d = self.D.get_data_descriptor('i_sense')
        d.multiply = 2.0
        self.D.set_data_descriptor(d, 'i_sense')
        mod_isense = self.D.get_data('i_sense')
        orig_isense = self.D.get_orig_data('i_sense')
        self.assertTrue(np.all(orig_isense == mod_isense / 2.0))

    def test_modifiy_data(self):
        self.D.initialize_minimum()
        d = self.D.get_data_descriptor('x')
        odata = d.data[:]
        d.multiply = 2.0
        d.offset = 10
        self.D.set_data_descriptor(d)
        ndata = self.D.get_data('x')
        self.assertTrue(np.allclose(odata, (ndata / 2.0 - 10), atol=1e-5))

    def test_set_data(self):
        self.D.initialize_minimum()
        data = np.ones((100, 1000, 10))
        self.D.set_data(data, 'test_data')
        get_data = self.D.get_data('test_data')
        self.assertTrue(np.allclose(data, get_data))

    def test_clear_caches(self):
        self.D.clear_caches()
        self.assertTrue(True)

    def test__initialize_minimum(self):
        self.D.initialize_minimum()
        self.assertTrue(self.D.initialized)


if __name__ == '__main__':

    class Test:
        def __init__(self, a: int, b: int):
            self.a = a
            self.b = b

        def __hash__(self):
            return hash(self.a)

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return self.a == other.a
            return False


    A = Test(1, 2)
    B = Test(1, 3)
    C = Test(2, 2)

    print(B in [A, C])
