from unittest import TestCase
from dat_analysis.plotting.mpl import util
import numpy as np


class Test(TestCase):
    def test_xyz_to_explicit_meshgrid(self):
        # Make 3x3 array
        x = np.array([1, 4, 10])
        y = np.array([5, 6, 6.5])
        z = x * y.T[:, None]

        # Convert to mpl plottable meshgrid (x, y specify corners for z instead of center coords)
        nx, ny, nz = util.xyz_to_explicit_meshgrid(x, y, z)

        # Drops the outermost data to avoid extrapolation
        exp_x = np.array([[2.5, 7.],
                          [2.5, 7.]])
        exp_y = np.array([[5.5, 5.5],
                          [6.25, 6.25]])
        exp_z = np.array([[24.]])  # Only expect the centre of 3x3 to remain
        self.assertTrue(np.all(nx == exp_x))
        self.assertTrue(np.all(ny == exp_y))
        self.assertTrue(np.all(nz == exp_z))
