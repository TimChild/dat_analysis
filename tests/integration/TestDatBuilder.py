from unittest import TestCase
from src.DatBuilder.Exp_to_standard import make_dat
from src.ExperimentSpecific.TestingDir.TestingESI import TestingESI
from src import CoreUtil as CU
import os
import logging

CU.set_default_logging()


class TestBuild(TestCase):
    datnum = 2711
    datname = 'base'
    esi = TestingESI(datnum)

    def test_overwrite_fit(self):
        dat = make_dat(TestBuild.datnum, TestBuild.datname, overwrite=True, dattypes=None,
                       ESI_class=TestingESI, run_fits=True)
        dat.hdf.close()
        del dat
        self.assertTrue(True)  # Just check it builds without errors

    def test_load(self):
        esi = TestBuild.esi
        path = esi.get_HDF_path(name=TestBuild.datname)
        assert os.path.isfile(path)  # Make sure it does exist before trying to load
        dat = make_dat(TestBuild.datnum, 'base', overwrite=False, dattypes=None,
                       ESI_class=TestingESI)  # Overwrite = False means load because file should exist
        dat.hdf.close()
        del dat
        self.assertTrue(True)  # Just check it builds without errors


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    dat = make_dat(2711, 'base', overwrite=False, dattypes=None,
                   ESI_class=TestingESI, run_fits=True)
    # dat.Transition.run_row_fits()
    # dat.Transition.run_avg_fit()
    #
    # centers = [fit.best_values.mid for fit in dat.Transition.all_fits]
    # dat.Entropy.recalculate_entr(centers)
    # dat.Entropy.run_row_fits()
    # dat.Entropy.run_avg_fit()
