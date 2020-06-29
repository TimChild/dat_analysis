from unittest import TestCase
from src.DatObject.Make_Dat import make_dat
from src.DataStandardize.ExpSpecific.Jun20 import JunESI
from src import CoreUtil as CU
import os

CU.set_default_logging()


class TestAWG(TestCase):
    datnum = 134
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


    # data = dat.Data.Exp_test_0_2d[0]
    data = dat.Data.Exp_wave2_2d[0]
    x = dat.Data.x_array

    fig, ax = plt.subplots(1)
    # ax.plot(x, data)

    fw = dat.AWG.get_full_wave(0)
    mws = dat.AWG.get_full_wave_masks(0)
    mw0 = mws[0]
    mwp = mws[1]
    mwm = mws[2]
    ax.plot(x, mw0 * data, label='0')
    ax.plot(x, mwp * data, label='p')
    ax.plot(x, mwm * data, label='m')

    hxs, harm1 = dat.AWG.get_per_cycle_harmonic(0, 1, data, x, skip_x=0)
    _, harm2 = dat.AWG.get_per_cycle_harmonic(0, 2, data, x, skip_x=0)

    ax.plot(hxs, harm1, label='harm1')
    ax.plot(hxs, harm2, label='harm2')
    ax.legend()