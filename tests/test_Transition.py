from unittest import TestCase
from src.DatObject.Attributes.Transition import Transition, DEFAULT_PARAMS, i_sense
from src.HDF_Util import with_hdf_read
from tests import helpers
import numpy as np
import lmfit as lm

output_dir = 'Outputs/Transition/'


class Testing_Transition(Transition):
    """Override the normal init behaviour so it doesn't fail before reaching tests"""

    # def __init__(self, dat):
    #
    #     super().__init__(dat)
    #     # TODO: Check that getting all FitPaths is not slow!
    #     # self.fit_paths = self._get_FitPaths()  # Container for different ways to look at fit paths
    #     self._avg_x = None
    #     self._avg_data = None
    #     self._avg_data_std = None
    #     self._avg_fit = None
    #     self._row_fits = None
    #
    # @with_hdf_read
    # def check_init(self):
    #     group = self.hdf.get(self.group_name, None)
    #     if group is None:
    #         self._create_group(self.group_name)
    #         group = self.hdf.get(self.group_name)
    #     if group.attrs.get('initialized', False) is False:
    #         # self._initialize()  # This will run everything otherwise
    #         pass


class TestTransition(TestCase):
    helpers.clear_outputs(output_dir)
    dat = helpers.init_testing_dat(9111, output_directory=output_dir)
    T = Testing_Transition(dat)

    def test_get_default_params(self):
        default_pars = self.T.get_default_params()
        self.assertEqual(default_pars, DEFAULT_PARAMS)

    def test_get_non_default_params(self):
        self.T.initialize_minimum()
        pars = self.T.get_default_params(self.T.x, self.T.data[0:5])
        self.assertTrue(np.all([isinstance(p, lm.Parameters) for p in pars]))

    def test_get_default_func(self):
        self.assertEqual(self.T.get_default_func(), i_sense)

    def test_initialize_minimum(self):
        self.T.initialize_minimum()
        self.assertTrue(self.T.initialized)


