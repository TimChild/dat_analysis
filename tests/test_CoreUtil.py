from unittest import TestCase
import numpy as np
import src.CoreUtil as CU


# class Test_MyLRU(TestCase):
#     class Temp:
#         def __init__(self):
#             self.x = 0
#
#         @CU.MyLRU
#         def set_var(self, a, b):
#             self.x += 1
#             return a, b
#
#     def setUp(self):
#         self.Temp.set_var.cache_clear()
#         self.T = self.Temp()
#
#     def test_run_1st_time(self):
#         x_before = self.T.x
#         a, b = self.T.set_var(1, 2)
#         x_after = self.T.x
#         self.assertEqual((1, 2), (a, b))
#         self.assertEqual(x_before, x_after - 1)  # i.e. it DID run through method
#
#     def test_cache_2nd_time(self):
#         self.T.set_var(1, 2)
#         x_before = self.T.x
#         a, b = self.T.set_var(1, 2)
#         x_after = self.T.x
#         self.assertEqual((a, b), (1, 2))
#         self.assertEqual(x_before, x_after)  # i.e. it did NOT run through method on second call
#
#     def test_cache_multiple(self):
#         self.T.set_var(1, 2)
#         self.T.set_var(3, 4)
#         x_before = self.T.x
#         first = self.T.set_var(1, 2)
#         second = self.T.set_var(3, 4)
#         x_after = self.T.x
#         self.assertEqual(x_before, x_after)
#         self.assertEqual(first, (1, 2))
#         self.assertEqual(second, (3, 4))
#
#     def test_cache_replace(self):
#         self.T.set_var(1, 2)
#         self.T.set_var.cache_replace((5, 6), 1, 2)
#         replaced = self.T.set_var(1, 2)
#         self.assertEqual((5, 6), replaced)
#
#     def test_cache_clear(self):
#         self.T.set_var(1, 2)
#         self.T.set_var.cache_clear()
#         x_before = self.T.x
#         self.T.set_var(1, 2)
#         x_after = self.T.x
#         self.assertEqual(x_before, x_after - 1)  # i.e. it DID run again
#
#     def test_cache_remove(self):
#         self.T.set_var(1, 2)
#         self.T.set_var(3, 4)
#         self.T.set_var.cache_remove(3, 4)
#         x_before = self.T.x
#         self.T.set_var(1, 2)
#         x_after = self.T.x
#         self.assertEqual(x_before, x_after)  # i.e. did not run
#
#         x_before = self.T.x
#         self.T.set_var(3, 4)
#         x_after = self.T.x
#         self.assertEqual(x_before, x_after - 1)  # i.e. did run
#
#     def test_independent_between_instances(self):
#         self.T.set_var(1, 2)
#         self.T.set_var.cache_replace((5), 1, 2)
#
#         N = self.Temp()
#         ans = N.set_var(1, 2)
#         self.assertEqual((1, 2), ans)


class Test(TestCase):
    def test_bin_data_1d(self):
        data = np.linspace(0, 100, 10000)
        binned = CU.bin_data_new(data, bin_x=10)
        print(data.shape)
        expected = np.linspace(0.045, 99.955, 1000)
        self.assertTrue(np.allclose(expected, binned, atol=0.0001))

    def test_bin_data_1d_non_multiple(self):
        data = np.linspace(0, 100, 10000)
        binned = CU.bin_data_new(data, bin_x=13)
        self.assertEqual((769,), binned.shape)

    def test_bin_dat_2d_big(self):
        x = np.linspace(0, 100, 1000)
        y = x
        xx, yy = np.meshgrid(x, y)
        data = np.cos(xx) + np.sin(yy)
        binned = CU.bin_data_new(data, bin_x=5, bin_y=10)
        self.assertEqual((100, 200), binned.shape)

    def test_bin_dat_2d(self):
        """Check that an equal amount is chopped off of each side before binning
        i.e. want to bin in groups of 3, that wont work with 11 long, so chop one off the beginning and one
        off of the end in x an y"""
        x = np.linspace(1, 11, 11)
        y = x
        xx, yy = np.meshgrid(x, y)
        data = xx * yy
        binned = CU.bin_data_new(data, bin_x=3, bin_y=3)
        self.assertTrue(np.allclose(np.array([[[9., 18., 27.], [18., 36., 54.], [27., 54., 81.]]]), binned))

    def test_bin_dat_3d(self):
        data = np.ones((100, 100, 100))
        binned = CU.bin_data_new(data, 3, 4, 5)
        self.assertEqual((20, 25, 33), binned.shape)
