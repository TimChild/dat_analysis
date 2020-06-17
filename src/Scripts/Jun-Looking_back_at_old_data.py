from src.Scripts.StandardImports import *

from src.ExperimentSpecific import Jan20Config

sep_datdf, sep_sf = IDD.get_exp_df('sep19', dfname='Jun20')
mar_datdf, mar_sf = IDD.get_exp_df('mar19', dfname='Jun20')
jan_datdf, jan_sf = IDD.get_exp_df('jan20', dfname='Jun20')
jun_datdf, jun_sf = IDD.get_exp_df('jun20', dfname='Jun20')


import scipy.signal


import lmfit
import re



if __name__ == '__main__':
    # dat = C.make_dat(1492, 'base', dfoption='overwrite', datdf=jan_datdf, setupdf=jan_sf, config=Jan20Config)
    # DF.update_save(dat, update=True, save=True, datdf=jan_datdf)
    dat = C.DatHandler.get_dat(1492, 'base', jan_datdf, jan_sf, Jan20Config)

    # pars = dat.Transition._avg_full_fit.params
    # import h5py
    # import os
    # from src.HDF import Util as U
    # os.remove('test.h5')
    # hdf = h5py.File('test.h5', 'a')
    # params_group = hdf.create_group('Params')
    # U.params_to_HDF(pars, params_group)
    # new_pars = U.params_from_HDF(params_group)
    # hdf.close()

    # data, x = CU.remove_nans(dat.Transition._avg_data, dat.Transition._x_array)
    # best_fit = dat.Transition._avg_full_fit.best_fit
    # deviation = data - fit.best_fit
    # fig, ax = plt.subplots(1)
    # # PF.Plots.deviation_from_fit(x, data, best_fit, ax)
    # deviation = data-best_fit
    # PF.Plots.power_spectrum(deviation, 2008/3, 1, ax, label='Average_filtered')
    #
    # data, x = dat.Data.ADC0_2d[0], dat.Data.x_array
    # best_fit = dat.Transition._full_fits[0].best_fit
    #
    # deviation = data-best_fit
    # PF.Plots.power_spectrum(deviation, 2008/3, 1, ax, label='Single_unfiltered')
    # ax.legend()

    # dat = C.DatHandler.get_dat(50, 'base')



