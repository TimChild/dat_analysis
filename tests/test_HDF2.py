from unittest import TestCase
from helpers import get_testing_Exp2HDF
from dat_analysis.dat_object.make_dat import DatHandler
import dat_analysis.dat_object.make_dat
import os
import shutil
dat_dir = os.path.abspath('fixtures/dats/2020Sep')
"""
Contents of dat_dir relevant in this file:
    Dat9111: Square entropy with dS ~Ln2
    
"""
# Where to put outputs (i.e. DatHDFs)
output_dir = os.path.abspath('Outputs/test_HDF2')
Testing_Exp2HDF = get_testing_Exp2HDF(dat_dir, output_dir)
dat_analysis.dat_object.make_dat.default_Exp2HDF = None  # Set default in Make_Dat (this is the most common use case)


class TestOpening(TestCase):
    """Testing basic opening of files"""

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    def test_a_init_dat_from_exp_data(self):
        dat = DatHandler().get_dat(9111, overwrite=False, init_level='min', exp2hdf=Testing_Exp2HDF)
        self.assertTrue(True)
        return dat

    def test_b_overwrite_dat(self):
        dat = self.test_a_init_dat_from_exp_data()
        DatHandler().get_dat(9111, overwrite=True, init_level='min', exp2hdf=Testing_Exp2HDF)
        self.assertTrue(True)

    def test_c_use_default_exp2hdf(self):
        """Do we get the same dat when relying on the set default Exp2HDF"""
        dat_analysis.dat_object.make_dat.default_Exp2HDF = Testing_Exp2HDF  # Set default in Make_Dat (this is the most common use case)
        DatHandler().get_dat(9111)
        self.assertTrue(True)
        dat_analysis.dat_object.make_dat.default_Exp2HDF = None  # Set default in Make_Dat (this is the most common use case)

    def test_d_basic_dat_attrs(self):
        dat = DatHandler().get_dat(9111, exp2hdf=Testing_Exp2HDF)
        exp_ans = {
            9111: dat.datnum,
            'base': dat.datname,
            'Dat9111': dat.dat_id,

        }
        for k, v in exp_ans.items():
            self.assertEqual(k, v)



if __name__ == '__main__':
    pass

