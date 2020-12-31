from unittest import TestCase
from src.DatObject.Attributes.AWG import AWG
from src.DatObject.Attributes.Logs import AWGtuple
from tests import helpers
from src.HDF_Util import NotFoundInHdfError
import numpy as np

output_dir = 'Outputs/AWG/'


class TestAWG(TestCase):
    helpers.clear_outputs(output_dir)
    dat = helpers.init_testing_dat(9111, output_directory=output_dir)
    A = AWG(dat)

    def tearDown(self) -> None:
        """Runs AFTER every test
        Check that HDF is left closed
        """
        with self.assertRaises(ValueError):
            filename = self.dat.hdf.hdf.filename  # Checking hdf is actually closed

    def test_info(self):
        info = self.A.info
        print(info)
        expected = AWGtuple(outputs={0: [0], 1: [1]}, wave_len=492, num_adcs=1, samplingFreq=6060.6, measureFreq=6060.6,
                            num_cycles=1, num_steps=148)
        for k in expected._fields:
            self.assertEqual(getattr(expected, k), getattr(info, k))

    def test_aws(self):
        aws = self.A.AWs
        print(aws)
        expected = (np.array([[-416.66666, -333.33334, -250., -166.66667, -83.333336,
                               0., 83.333336, 166.66667, 250., 333.33334,
                               416.66666, 500., 416.66666, 333.33334, 250.,
                               166.66667, 83.333336, 0., -83.333336, -166.66667,
                               -250., -333.33334, -416.66666, -500.],
                              [4., 4., 4., 4., 4.,
                               103., 4., 4., 4., 4.,
                               4., 103., 4., 4., 4.,
                               4., 4., 103., 4., 4.,
                               4., 4., 4., 103.]]),
                    np.array([[65.375, 52.3, 39.225, 26.15, 13.075, 0., -13.075,
                               -26.15, -39.225, -52.3, -65.375, -78.45, -65.375, -52.3,
                               -39.225, -26.15, -13.075, 0., 13.075, 26.15, 39.225,
                               52.3, 65.375, 78.45],
                              [4., 4., 4., 4., 4., 103., 4.,
                               4., 4., 4., 4., 103., 4., 4.,
                               4., 4., 4., 103., 4., 4., 4.,
                               4., 4., 103.]]))
        self.assertTrue(all([np.allclose(exp, aw) for exp, aw in zip(expected, aws.values())]))

    def test_freq(self):
        print(self.A.freq)
        self.assertTrue(np.isclose(12.318, self.A.freq, atol=0.001))

    def test_measure_freq(self):
        print(self.A.measure_freq)
        self.assertEqual(6060.6, self.A.measure_freq)

    def test_numpts(self):
        print(self.A.numpts)
        self.assertEqual(72816, self.A.numpts)

    def test_true_x_array(self):
        print(self.A.true_x_array)
        expected = np.linspace(-276.57000732, 323.42999268, 148)
        self.assertTrue(np.allclose(expected, self.A.true_x_array))

    def test_get_single_wave(self):
        expected = [-416.66665649, -416.66665649, -416.66665649, -416.66665649, -333.33334351,
                    -333.33334351, -333.33334351, -333.33334351, -250., -250.,
                    -250., -250., -166.66667175, -166.66667175, -166.66667175,
                    -166.66667175, -83.33333588, -83.33333588, -83.33333588, -83.33333588,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 0., 0,
                    0., 0., 0., 83.33333588, 83.33333588,
                    83.33333588, 83.33333588, 166.66667175, 166.66667175, 166.66667175,
                    166.66667175, 250., 250., 250., 250.,
                    333.33334351, 333.33334351, 333.33334351, 333.33334351, 416.66665649,
                    416.66665649, 416.66665649, 416.66665649, 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 500., 500., 500., 500.,
                    500., 416.66665649, 416.66665649, 416.66665649, 416.66665649,
                    333.33334351, 333.33334351, 333.33334351, 333.33334351, 250.,
                    250., 250., 250., 166.66667175, 166.66667175,
                    166.66667175, 166.66667175, 83.33333588, 83.33333588, 83.33333588,
                    83.33333588, 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., -83.33333588,
                    -83.33333588, -83.33333588, -83.33333588, -166.66667175, -166.66667175,
                    -166.66667175, -166.66667175, -250., -250., -250.,
                    -250., -333.33334351, -333.33334351, -333.33334351, -333.33334351,
                    -416.66665649, -416.66665649, -416.66665649, -416.66665649, -500.,
                    -500., -500., -500., -500., -500,
                    -500., -500., -500., -500., -500,
                    -500., -500., -500., -500., -500,
                    -500., -500., -500., -500., -500,
                    -500., -500., -500., -500., -500,
                    -500., -500., -500., -500., -500,
                    -500., -500., -500., -500., -500,
                    -500., -500., -500., -500., -500,
                    -500., -500., -500., -500., -500,
                    -500., -500., -500., -500., -500,
                    -500., -500., -500., -500., -500,
                    -500., -500., -500., -500., -500,
                    -500., -500., -500., -500., -500,
                    -500., -500., -500., -500., -500,
                    -500., -500., -500., -500., -500,
                    -500., -500., -500., -500., -500,
                    -500., -500., -500., -500., -500,
                    -500., -500., -500., -500., -500,
                    -500., -500., -500., -500., -500.,
                    -500., -500., -500., -500., -500.,
                    -500., -500.]
        self.assertTrue(np.allclose(expected, self.A.get_single_wave(0)))

        print(self.A.get_single_wave(1).shape)
        self.assertTrue(492, self.A.get_single_wave(1).shape)

        with self.assertRaises(NotFoundInHdfError):
            self.A.get_single_wave(3)

    def test_get_full_wave(self):
        print(self.A.get_full_wave(0).shape)

        self.assertEqual((72816,), self.A.get_full_wave(0).shape)

        with self.assertRaises(NotFoundInHdfError):
            self.A.get_full_wave(3)

    def test_get_single_wave_masks(self):
        print(self.A.get_single_wave_masks(0).shape)
        self.assertEqual((24, 492), self.A.get_single_wave_masks(0).shape)

        with self.assertRaises(NotFoundInHdfError):
            self.A.get_single_wave_masks(3)

    def test_get_full_wave_masks(self):
        print(self.A.get_full_wave_masks(0).shape)
        self.assertTrue(np.allclose((24, 72816), self.A.get_full_wave_masks(0).shape))

        with self.assertRaises(NotFoundInHdfError):
            self.A.get_full_wave_masks(3)

    def test_eval(self):
        for x, expected in zip([-276.57001, -275.33398, -274.41110, 323.42999], [-416.7, 500, 83.3, -500]):
            print(self.A.eval(x, wave_num=0))
            self.assertTrue(np.isclose(expected, self.A.eval(x, wave_num=0), atol=0.1))
