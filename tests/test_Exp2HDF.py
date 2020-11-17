from unittest import TestCase
from tests.helpers import get_testing_Exp2HDF
from src.DataStandardize import BaseClasses
from src.DataStandardize.ExpConfig import ExpConfigBase
import os
dat_dir = os.path.abspath(r'fixtures\dats\2020Sep\\')
"""
Contents of dat_dir relevant in this file:
    Dat9111: Square entropy with dS ~Ln2

"""

# Where to put outputs (i.e. DatHDFs)
output_dir = 'Outputs/test_DatHDFBuilder/'


Modified_Exp2HDF = get_testing_Exp2HDF(dat_dir, output_dir)


class Test_Exp2HDF(TestCase):
    """Checking Exp2HDF"""

    def test_init_1(self):
        exp2hdf = Modified_Exp2HDF(1)
        self.assertTrue(True)
        return exp2hdf

    def test_init_10000(self):
        exp2hdf = Modified_Exp2HDF(10000)
        del exp2hdf
        self.assertTrue(True)

    def test_get_ExpConfig(self):
        exp2hdf = self.test_init_1()
        exp_config = exp2hdf.ExpConfig
        self.assertIsInstance(exp_config, ExpConfigBase)
        return exp_config

    def test_get_SysConfig(self):
        exp2hdf = self.test_init_1()
        sys_config = exp2hdf.SysConfig
        self.assertIsInstance(sys_config, BaseClasses.SysConfigBase)
        return sys_config
