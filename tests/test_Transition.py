from unittest import TestCase
import time
import shutil
import os
from src.dat_object.attributes.transition import Transition, default_transition_params, i_sense
from src.dat_object.dat_hdf import DatHDF
import h5py
from src.hdf_util import with_hdf_read
from tests import helpers
import numpy as np
import lmfit as lm

output_dir = 'Outputs/Transition/'


class Testing_Transition(Transition):
    """Override the normal init behaviour so it doesn't fail before reaching tests"""


class TestTransition(TestCase):
    helpers.clear_outputs(output_dir)
    dat = helpers.init_testing_dat(9111, output_directory=output_dir)
    T = Testing_Transition(dat)

    def test_get_default_params(self):
        default_pars = self.T.get_default_params()
        self.assertEqual(default_pars, default_transition_params())

    def test_get_non_default_params(self):
        self.T.initialize_minimum()
        pars = self.T.get_default_params(self.T.x, self.T.data[0:5])
        self.assertTrue(np.all([isinstance(p, lm.Parameters) for p in pars]))

    def test_get_default_func(self):
        self.assertEqual(self.T.get_default_func(), i_sense)

    def test_initialize_minimum(self):
        self.T.initialize_minimum()
        self.assertTrue(self.T.initialized)


class TestExistingTransition(TestCase):
    def setUp(self):
        out_path = 'Outputs/Transition/DatHDFs[Transition].h5'
        if os.path.exists(out_path):
            os.remove(out_path)
        shutil.copy2('fixtures/DatHDFs/Dat9111[Transition].h5', out_path)
        self.dat = DatHDF(h5py.File(out_path, 'r'))  # A dat with Transition info already filled
        self.t0 = time.time()

    def test_load_avg_fit(self):
        """Check that getting an existing avg fit is fast"""
        fit = self.dat.Transition.avg_fit
        self.assertLess(time.time()-self.t0, 1)  # Should take less than 1 second to retrieve fit from HDF

    def test_load_avg_data(self):
        """Check that getting existing avg data is fast"""
        data =  self.dat.Transition.avg_data
        self.assertLess(time.time()-self.t0, 1)  # Should take less than 1 second to retrieve data from HDF
