from unittest import TestCase
from src.DatObject.Attributes.Entropy import Entropy, DEFAULT_PARAMS, entropy_nik_shape
from tests import helpers
import numpy as np
import lmfit as lm
import shutil
from src.DatObject.DatHDF import DatHDF
import time
import os
import h5py

output_dir = 'Outputs/Entropy/'


class TestEntropy(TestCase):
    helpers.clear_outputs(output_dir)
    dat = helpers.init_testing_dat(9111, output_directory=output_dir)
    E = Entropy(dat)

    def test_get_default_params(self):
        default_pars = self.E.get_default_params()
        self.assertEqual(DEFAULT_PARAMS, default_pars)
        self.assertTrue(True)

    def test_get_non_default_params(self):
        pars = self.E.get_default_params(self.E.x, self.E.data[0:5])
        self.assertTrue(np.all([isinstance(p, lm.Parameters) for p in pars]))

    def test_get_default_func(self):
        self.assertEqual(entropy_nik_shape, self.E.get_default_func())

    def test_get_avg_entropy(self):
        fit = self.E.avg_fit
        print(fit.best_values)
        expected = (0.7149, 0.0736, -0.969, 16.384)
        bv = fit.best_values
        self.assertTrue(np.allclose(expected, (bv.dS, bv.dT, bv.mid, bv.theta), rtol=0.01, atol=0.01))


class TestExistingEntropy(TestCase):
    def setUp(self):
        out_path = 'Outputs/Entropy/DatHDFs[Entropy].h5'
        if os.path.exists(out_path):
            os.remove(out_path)
        shutil.copy2('fixtures/DatHDFs/Dat9111[Entropy].h5', out_path)
        self.dat = DatHDF(h5py.File(out_path, 'r'))  # A dat with Transition info already filled
        self.t0 = time.time()

    def test_load_avg_fit(self):
        """Check that getting an existing avg fit is fast"""
        fit = self.dat.Entropy.avg_fit
        self.assertLess(time.time()-self.t0, 1)  # Should take less than 1 second to retrieve fit from HDF

    def test_load_avg_data(self):
        """Check that getting existing avg data is fast"""
        data = self.dat.Entropy.avg_data
        self.assertLess(time.time()-self.t0, 1)  # Should take less than 1 second to retrieve data from HDF
