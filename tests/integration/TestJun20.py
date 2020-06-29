from unittest import TestCase
from src.DatObject.Make_Dat import make_dat
from src.DataStandardize.ExpSpecific.Jun20 import JunESI
from src import CoreUtil as CU
import os

CU.set_default_logging()


class TestAWG(TestCase):
    datnum = 133
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

    fig, ax = plt.subplots(1)
    data = dat.Data.Exp_test_0_2d[0]
    ax.plot(dat.Data.x_array, data)

    x = dat.Data.x_array
    fw = dat.AWG.get_full_wave(0)
    mws = dat.AWG.get_full_wave_masks(0)
    mw0 = mws[0]
    mwp = mws[1]
    mwm = mws[2]
    ax.plot(x, mw0 * fw)
    ax.plot(x, mwp * fw)
    ax.plot(x, mwm * fw)

    # per cycle 1st harmonic
    aw = dat.AWG.AWs[0]
    i = 0
    wl = dat.AWG.info.wave_len
    chunk = data[i*wl:i*wl+int(aw[1][0])]

    harm1 = []
    harm2 = []
    for i in range(dat.AWG.info.num_cycles):
        a0 = np.nanmean(fw[i*wl:(i+1)*wl]*mw0[i*wl:(i+1)*wl])
        ap = np.nanmean(fw[i*wl:(i+1)*wl]*mwp[i*wl:(i+1)*wl])
        am = np.nanmean(fw[i*wl:(i+1)*wl]*mwm[i*wl:(i+1)*wl])
        h1 = ((ap-a0)+(a0-am))/2
        h2 = ((ap-a0)+(am-a0))/2
        harm1.append(h1)
        harm2.append(h2)
    hxs = np.linspace(x[round(wl/2)], x[-round(wl/2)], dat.AWG.info.num_cycles)

    ax.plot(hxs, harm1, label='harm1')
    ax.plot(hxs, harm2, label='harm2')
    ax.legend()