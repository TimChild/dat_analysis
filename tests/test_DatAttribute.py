from unittest import TestCase
import lmfit as lm
import h5py
import copy
from src.DatObject.Attributes.DatAttribute import DataDescriptor, FitPaths
from src.AnalysisTools.fitting import FitInfo, FitIdentifier
import numpy as np
from src.DatObject.Attributes import Transition
from tests import helpers
from src.HDF_Util import with_hdf_read, NotFoundInHdfError

output_dir = 'Outputs/DatAttribute/'


class Testing_Transition(Transition.Transition):
    """Override the normal init behaviour so it doesn't fail before reaching tests"""

    def __init__(self, dat):
        super().__init__(dat)

    @with_hdf_read
    def check_init(self):
        group = self.hdf.get(self.group_name, None)
        if group is None:
            self._create_group(self.group_name)
            group = self.hdf.get(self.group_name)
        if group.attrs.get('initialized', False) is False:
            # self._initialize()  # This will run everything otherwise

            # At least run this which just set's groups for fit attrs (TESTING ONLY)
            self._set_default_fit_groups()


class TestDatAttributeWithData(TestCase):
    helpers.clear_outputs(output_dir)
    dat = helpers.init_testing_dat(9111, output_dir)
    T: Transition.Transition = Testing_Transition(dat)

    def test_get_data(self):
        data = self.T.get_data('x')
        print(data)
        expected = [-276.57, -276.56177, -276.55353]
        self.assertTrue(np.allclose(expected, data[:3]))

    def test_set_data(self):
        data = np.linspace(0, 100, 1000)
        self.T.set_data('test_data', data)
        get_data = self.T.get_data('test_data')
        self.assertTrue(np.allclose(data, get_data))

    def test_specific_data_descriptors(self):
        sdd = self.T.specific_data_descriptors_keys
        print(sdd)
        self.assertIsInstance(sdd['test_data'], DataDescriptor)

    def test_set_data_descriptor(self):
        orig = self.T.get_descriptor('x')
        new = copy.copy(orig)
        new.multiply = 2.0
        self.T.set_data_descriptor(new, 'test_descriptor')
        new_get = self.T.get_descriptor('test_descriptor')
        self.assertEqual(new, new_get)

    def test_get_descriptor(self):
        desc = self.T.get_descriptor('x')
        print(desc)
        expected = DataDescriptor('/Experiment Copy/x_array')
        self.assertEqual(expected, desc)


class TestFittingAttribute(TestCase):
    helpers.clear_outputs(output_dir)
    dat = helpers.init_testing_dat(9111, output_dir)
    T: Transition.Transition = Testing_Transition(dat)

    def tearDown(self):
        """Runs AFTER every test"""
        with self.assertRaises(ValueError):
            print(self.dat.hdf.hdf.filename)  # Checking hdf is actually closed

    def test_get_default_params(self):
        default_pars = self.T.get_default_params()
        self.assertEqual(Transition.default_transition_params(), default_pars)

    def test_get_default_func(self):
        func = self.T.get_default_func()
        self.assertEqual(Transition.i_sense, func)

    def test_default_data_names(self):
        names = self.T.default_data_names()
        print(names)
        expected = ['x', 'i_sense']
        self.assertEqual(expected, names)

    def test_clear_caches(self):
        self.T.clear_caches()
        self.assertTrue(True)

    def test_get_centers(self):
        centers = self.T.get_centers()
        expected = [28.945195081474278, 30.14704210128557, 18.172868224059584, 13.886815652413931, 15.613973966508519]
        close = [np.isclose(e, r, atol=0.01, rtol=0.001) for e, r in zip(expected, centers[0:5])]
        print(close)
        self.assertTrue(all(close))

    def test_avg_data(self):
        avg = self.T.avg_data
        self.assertEqual(70842, sum(~np.isnan(avg)))

    def test_avg_x(self):
        avg_x = self.T.avg_x
        print(avg_x)
        expected = [-296.65248355, -296.64424349, -296.63600343]
        self.assertTrue(np.allclose(expected, avg_x[:3], atol=0.00001))

    def test_avg_data_std(self):
        avg_data_std = self.T.avg_data_std
        print(avg_data_std)
        expected = [0.02174817, 0.02046346, 0.01803641]
        self.assertTrue(np.allclose(expected, avg_data_std[0:3], atol=0.0001))

    def test_avg_fit(self):
        avg_fit = self.T.avg_fit
        self.assertAlmostEqual(13.7, avg_fit.best_values.theta, delta=0.1)

    def test_row_fits(self):
        row_fits = self.T.row_fits
        for f in row_fits:
            self.assertIsInstance(f.best_values.mid, float)

    def test_get_avg_data(self):
        avg_data = self.T.get_avg_data(name='testavg')
        avd2, avg_std, avg_x = self.T.get_avg_data(return_std=True, return_x=True, name='testavg')
        self.assertTrue(np.allclose(avg_data, avd2, atol=0.0001, equal_nan=True))
        self.assertIsInstance(avg_std, np.ndarray)

    def test__make_avg_data(self):
        self.test_get_avg_data()
        self.assertTrue(True)

    def test_get_fit_make(self):
        fit = self.T.get_fit(which='row', row=10, name='test_fit', check_exists=False)
        print(fit.best_values)
        expected = 19.57
        self.assertTrue(np.isclose(expected, fit.best_values.mid, atol=0.01))
        return fit

    def test_get_fit(self):
        ofit = self.test_get_fit_make()
        fit = self.T.get_fit(which='row', row=10, name='test_fit', check_exists=True)
        self.assertEqual(ofit, fit)
        with self.assertRaises(NotFoundInHdfError):
            self.T.get_fit(name='non_existing', check_exists=True)

    def test__get_fit_from_path(self):
        self.test_avg_fit()
        path = '/Transition/Avg Fits/default_avg'
        fit = self.T._get_fit_from_path(path)
        print(fit.best_values)
        self.assertAlmostEqual(13.7, fit.best_values.theta, delta=0.1)

        # Test on non existing path
        path = self.test__generate_fit_path()
        with self.assertRaises(NotFoundInHdfError):
            self.T._get_fit_from_path(path)

    def test_fit_paths(self):
        [self.T.get_fit('row', i, 'few_test_fits', check_exists=False) for i in range(5)]  # Generate a few fits
        # rows = self.T.row_fits  # To generate some fit paths if not already existing
        paths = self.T.fit_paths
        self.assertIsInstance(paths, FitPaths)
        rows = self.T.fit_paths.row_fits
        expected = '/Transition/Row Fits/0/few_test_fits_row[0]'
        self.assertEqual(expected, rows['few_test_fits_row[0]'])

    def test__get_fit_path_from_fit_id(self):
        params = Transition.default_transition_params()
        params['const'].value = 4.001  # So I know it doesn't match any other tests
        func = Transition.i_sense
        [self.T.get_fit('row', i, 'few_test_fits', initial_params=params, fit_func=func,
                        check_exists=False, overwrite=True) for i in range(2)]  # Generate a few fits
        # fit_path: str = list(self.T.fit_paths.all_fits.values())[0]
        # print(fit_path)
        # fit: FitInfo = self.T._get_fit_from_path(fit_path)
        # print(params, fit.params)
        params = Transition.default_transition_params()  # In case they were modified during fitting
        params['const'].value = 4.001  # To match same as before
        fit_id = FitIdentifier(params, Transition.i_sense, self.T.data[0])
        path = self.T._get_fit_path_from_fit_id(fit_id)
        self.assertEqual('/Transition/Row Fits/0/few_test_fits_row[0]', path)

    def test__get_fit_path_from_name(self):
        self.test_get_fit_make()
        path = self.T._get_fit_path_from_name('test_fit', which='row', row=10)
        print(path)
        expected = '/Transition/Row Fits/10/test_fit_row[10]'
        self.assertEqual(expected, path)

    def test__generate_fit_path(self):
        path = self.T._generate_fit_path(which='avg', name='new_fit_path')
        print(path)
        expected = '/Transition/Avg Fits/new_fit_path_avg'
        self.assertEqual(expected, path)
        return path

    def test__save_fit(self):
        fit = FitInfo()
        self.T._save_fit(fit, which='avg', name='test_save_fit')
        with h5py.File(self.T.hdf.hdf_path, 'r') as f:
            print('debug here to find where the file is saved')
            group = f.get('/Transition/Avg Fits/test_save_fit_avg')
            self.assertIsNotNone(group)

    def test__get_fit_parent_group_name(self):
        name = self.T._get_fit_parent_group_name(which='row', row=1)
        print(name)
        expected = '/Transition/Row Fits/1'
        name2 = self.T._get_fit_parent_group_name(which='avg')
        print(name2)
        expected2 = '/Transition/Avg Fits'
        self.assertEqual(expected, name)
        self.assertEqual(expected2, name2)

    def test__calculate_fit(self):
        x = np.linspace(0, 10, 10000)
        np.random.seed(0)
        data = (np.random.random(10000) * 2 - 1) + 2 * x + 5

        def line(x, a, c):
            return x * a + c

        params = lm.Parameters()
        params.add('a', 1)
        params.add('c', 0)
        fit = self.T._calculate_fit(x=x, data=data, params=params, func=line, auto_bin=True)
        print(fit.best_values)
        self.assertTrue(np.allclose((1.9993, 4.9964), (fit.best_values.a, fit.best_values.c), atol=0.001))

    def test_initialize_minimum(self):
        self.T.initialize_minimum()
        self.assertTrue(True)

    def test_set_default_data_descriptors(self):
        self.T.set_default_data_descriptors()
        desc = self.T.specific_data_descriptors_keys['i_sense']
        self.assertIsInstance(desc, DataDescriptor)

    def test__get_fit_saved_name(self):
        avg_name = self.T._generate_fit_saved_name('test', which='avg')
        self.assertEqual('test_avg', avg_name)
        row_name = self.T._generate_fit_saved_name('test', which='row', row=17)
        self.assertEqual('test_row[17]', row_name)
        with self.assertRaises(ValueError):
            a = self.T._generate_fit_saved_name('test', which='not_valid')


