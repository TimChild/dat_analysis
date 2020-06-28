from unittest import TestCase
from src.DatObject.Make_Dat import make_dat
from src.DataStandardize.ExpSpecific.Jun20 import JunESI
from src import CoreUtil as CU
import os

CU.set_default_logging()


class TestAWG(TestCase):
    datnum = 124
    datname = 'base'
    esi = JunESI(datnum)

    def test_overwrite(self):
        dat = make_dat(TestAWG.datnum, TestAWG.datname, overwrite=True, dattypes=None,
                       ESI_class=JunESI, run_fits=False)
        dat.hdf.close()
        del dat
        self.assertTrue(True)  # Just check it builds without errors

    def test_load(self):
        esi = TestAWG.esi
        path = esi.get_HDF_path(name=TestAWG.datname)
        assert os.path.isfile(path)  # Make sure it does exist before trying to load
        dat = make_dat(TestAWG.datnum, 'base', overwrite=False, dattypes=None,
                       ESI_class=JunESI)  # Overwrite = False means load because file should exist
        dat.hdf.close()
        del dat
        self.assertTrue(True)  # Just check it builds without errors


if __name__ == '__main__':
    from src.Scripts.StandardImports import *
    dat = make_dat(TestAWG.datnum, TestAWG.datname, overwrite=True, dattypes=None,
                   ESI_class=JunESI, run_fits=False)
    print('done')
    fig, ax = plt.subplots(1)
    data = dat.Data.Exp_test_0_2d[0]
    ax.plot(dat.Data.x_array, data)