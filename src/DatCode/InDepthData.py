import copy
import logging
import lmfit as lm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from typing import Union, List, Tuple

import src.PlottingFunctions as PF
import src.Core as C
import src.CoreUtil as CU
import src.Constants as Const
from src.DFcode import DatDF as DF, SetupDF as SF
import src.DatCode.Dat as D
from src.CoreUtil import sub_poly_from_data
from src.DatCode import Transition as T, Entropy as E

logger = logging.getLogger(__name__)


class DatMeta(object):
    """
    Object to store info about how to use data from given dataset.
    Optional to pass in dc info for entropy scans
    """

    def __init__(self, datnum=None, datname=None,
                 dc_num=None, dc_name=None,
                 datdf=None,
                 dt=None,
                 rows=None, row_remove=None,
                 bin_data_points=1,
                 i_spacing_y=-1, i_spacing_x=0,
                 e_spacing_y=1, e_spacing_x=0,
                 smoothing_num=1, alpha=None, i_func=None, config=None,
                 set_name=None):
        """
        Store information about how to use data from given dataset and which dcbias data goes along with it

        @param datnum: datnum of dataset
        @type datnum: Union[int, None]
        @param datname: datname of dataset
        @type datname: str
        @param dc_num: datnum of dcbias dataset
        @type dc_num: int
        @param dc_name: datname of dcbias data
        @type dc_name: str
        @param datdf: datDF to use to load dataset and dcbias
        @type datdf: DF.DatDF
        @param dt: dt to use for scaling if no dcbias dat provided
        @type dt: float
        @param rows: which rows of data to use from dataset
        @type rows: Union[None, list, set]
        @param row_remove: Which rows to remove from the loaded rows (self.rows)
        @type rows: Union[None, list, set]
        @param bin_data_points: Bin data into this size (useful to speed up calculations on dense datasets)
        @type bin_data_points: int
        @param i_spacing_y: relative spacing of charge sensor data in waterfall plot
        @type i_spacing_y: float
        @param i_spacing_x: relative spacing of charge sensor data in waterfall plot
        @type i_spacing_x: float
        @param e_spacing_y: relative spacing of entropy data in waterfall plot
        @type e_spacing_y: float
        @param e_spacing_x: relative spacing of entropy data in waterfall plot
        @type e_spacing_x: float
        @param smoothing_num: how much to smooth data for displaying fits (possibly needs to be odd)
        @type smoothing_num: int
        @param alpha: lever arm in SI units (i.e. T = alpha/kB * Theta, where T in K, theta in mV)
        @type alpha: float
        """
        self.set_name = set_name
        self.datnum = datnum
        self.datname = datname
        self.dc_num = dc_num
        self.dc_name = dc_name
        self.datdf = datdf
        self.dt = dt
        self.dat = self.set_dat(datnum, datname, datdf, verbose=False, config=config)
        self.dc = self.set_dc(dc_num, dc_name, datdf, verbose=False, config=config)
        self.i_func = i_func
        self.rows = rows  # rows to load initially
        self.row_remove = row_remove  # rows to remove from self.rows
        self.bin_data_points = bin_data_points  # How many data points to bin over
        self.i_spacing_y = i_spacing_y  # How much to spread out data in waterfall plots
        self.i_spacing_x = i_spacing_x
        self.e_spacing_y = e_spacing_y
        self.e_spacing_x = e_spacing_x
        self.smoothing_num = smoothing_num  # How many data points to smooth over
        self.alpha = alpha  # lever arm

        if self.row_remove is not None:  # make self.rows only contain desired rows
            self.set_rows()

    @property
    def dt(self):
        if self._dt is not None:
            return self._dt
        else:
            if None not in [self.dc, self.dat]:
                return self.dc.DCbias.get_dt_at_current(self.dat.Instruments.srs1.out / 50 * np.sqrt(2))
        print(f'No available "dt" for [{self.datnum}][{self.datname}]')
        return None

    @dt.setter
    def dt(self, value):
        self._dt = value

    @property
    def row_remove(self):
        return self._row_remove

    @row_remove.setter
    def row_remove(self, value):
        self._row_remove = value
        self.set_rows()

    def set_rows(self):
        if self.row_remove is not None:
            if self.rows is not None:
                self.rows = list(set(self.rows) - set(self.row_remove))
            else:
                self.rows = list(set(range(len(self.dat.Data.y_array))) - set(self.row_remove))

    def set_dat(self, datnum=None, datname=None, datdf=None, config=None, verbose=True):
        datnum = datnum if datnum is not None else self.datnum
        datname = datname if datname is not None else self.datname
        datdf = datdf if datdf is not None else self.datdf
        if None not in [datnum, datname, datdf]:
            self.datnum = datnum
            self.datname = datname
            self.datdf = datdf
            self.dat = C.DatHandler.get_dat(datnum, datname, datdf, config=config)
        else:
            logger.warning(f'More info required to set_dat: (datnum, datname, datdf) = \n'
                           f'[{datnum}, {datname}, {datdf}]')
            return None
        return self.dat

    def set_dc(self, dc_num=None, dc_name=None, datdf=None, config=None, verbose=True):
        datnum = dc_num if dc_num is not None else self.dc_num
        datname = dc_name if dc_name is not None else self.dc_name
        datdf = datdf if datdf is not None else self.datdf
        if None not in [datnum, datname, datdf]:
            self.dc_num = dc_num
            self.dc_name = dc_name
            self.datdf = datdf
            self.dc = C.DatHandler.get_dat(dc_num, dc_name, datdf, config=config)
        else:
            C.print_verbose(f'WARNING[DatMeta]: More info required to set_dc: (dc_num, dc_name, datdf) = \n'
                            f'[{datnum}, {datname}, {datdf}]', verbose)
            self.dc = None
        return self.dc


def get_dat_setup(datnum, set_name='', datdf=None):
    """
    Neatening up where I store all the dat setup info

    @param datnum: datnum to load
    @type datnum: int
    """

    # Make variables accessible outside this function

    if set_name.lower() == 'jan20':
        # [1533, 1501]
        if datdf is None:
            datdf = DF.DatDF(dfname='Apr20')
        if datnum == 1533:
            meta = DatMeta(1533, 'digamma_quad',
                           1529, 'base', datdf,
                           rows=list(range(0, 22, 5)),
                           bin_data_points=10,
                           i_spacing_y=-2,
                           e_spacing_y=-3,
                           smoothing_num=11,
                           alpha=CU.get_alpha(0.82423, 100),
                           i_func=T.i_sense_digamma_quad
                           )
        elif datnum == 1501:
            meta = DatMeta(1501, 'digamma_quad',
                           1529, 'base', datdf,
                           rows=list(range(22 - 6, 22, 1)),
                           bin_data_points=10,
                           i_spacing_y=-1,
                           e_spacing_y=-3,
                           smoothing_num=11,
                           alpha=CU.get_alpha(0.82423, 100),
                           i_func=T.i_sense_digamma_quad
                           )
        else:
            raise ValueError(f'setup data for [{datnum}] does not exist in set [{set}]')

    elif set_name.lower() == 'jan20_gamma':
        datnums = InDepthData.get_datnums(set_name)
        if datdf is None:
            datdf = DF.DatDF(dfname='Apr20')
        metas = [DatMeta(datnum=None, datname='base',
                         # Don't want to load all dats unless necessary because it's slow
                         dc_num=1529, dc_name='base', datdf=datdf,
                         rows=None,
                         row_remove=None,
                         bin_data_points=50,
                         i_spacing_y=-2,
                         i_spacing_x=-0,
                         e_spacing_y=-3,
                         e_spacing_x=0,
                         smoothing_num=1,
                         alpha=CU.get_alpha(0.82423, 100),
                         i_func=T.i_sense_digamma_quad,
                         config=None,
                         set_name='Jan20_1') for _ in datnums]
        metas = dict(zip(datnums, metas))
        if datnum == 1492:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {2, 6, 7, 9, 13, 14, 15, 21}
        elif datnum == 1495:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {0, 1, 3, 9, 13, 15}
        elif datnum == 1498:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {3, 4, 11}
        elif datnum == 1501:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {0, 3, 4}
        elif datnum == 1504:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {3, 6, 7}
        elif datnum == 1507:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {0, 4, 5, 6, 8}
        elif datnum == 1510:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {3, 4, 8, 10, 11}
        elif datnum == 1513:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {0, 2, 3, 4, 8, 10, 11}
        elif datnum == 1516:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {0, 3}
        elif datnum == 1519:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {3, 4, 8, 11}
        elif datnum == 1522:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {0, 1, 2, 8, 11}
        elif datnum == 1525:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {0, 1, 2, 4, 8, 9, 10, 11}
        elif datnum == 1528:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11}
        else:
            raise ValueError(f'setup data for [{datnum}] does not exist in set [{set_name}]')
        meta = metas[datnum]

    elif set_name.lower() == 'jan20_gamma_2':
        if datdf is None:
            datdf = DF.DatDF(dfname='Apr20')
        datnums = InDepthData.get_datnums(set_name)
        # Set defaults
        metas = [DatMeta(datnum=None, datname='base',
                         # Don't want to load all dats unless necessary because it's slow
                         dc_num=1529, dc_name='base', datdf=datdf,
                         rows=None,
                         row_remove=None,
                         bin_data_points=50,
                         i_spacing_y=-2,
                         e_spacing_y=-3,
                         smoothing_num=1,
                         alpha=CU.get_alpha(0.82423, 100),
                         i_func=T.i_sense_digamma_quad,
                         set_name='Jan20_2') for num in datnums]
        metas = dict(zip(datnums, metas))
        if datnum == 1533:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {0, 1, 3, 9, 20, 21}
        elif datnum == 1536:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {0, 16, 17}
        elif datnum == 1539:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {1, 2, 7, 14, 15}
        elif datnum == 1542:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {0, 1, 2, 5, 6, 9, 10}
        elif datnum == 1545:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {4, 6}
        elif datnum == 1548:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {3, 4, 8, 9, 11}
        elif datnum == 1551:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {3, 4, 5}
        elif datnum == 1554:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {0, 2, 4, 10}
        elif datnum == 1557:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {2, 4}
        elif datnum == 1560:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {1, 2, 3, 11}
        elif datnum == 1563:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {0, 1, 2, 11}
        elif datnum == 1566:
            metas[datnum].set_dat(datnum)
            metas[datnum].row_remove = {1, 3, 4, 5, 10, 11}
        else:
            raise ValueError(f'setup data for [{datnum}] does not exist in set [{set_name}]')
        meta = metas[datnum]

    elif set_name.lower() == 'sep19_gamma':
        if datdf is None:
            datdf = get_exp_df('sep19', dfname='Apr20')
        from src.Configs import Sep19Config
        datnums = InDepthData.get_datnums(set_name)
        if datnum in datnums:
            # Set defaults
            metas = [DatMeta(datnum=None, datname='base',
                             # Don't want to load all dats unless necessary because it's slow
                             dc_num=1945, dc_name='base', datdf=datdf,
                             rows=list(range(0, 21)),
                             row_remove=None,
                             bin_data_points=10,
                             i_spacing_y=-2,
                             i_spacing_x=-0.3,
                             e_spacing_y=-3,
                             e_spacing_x=0,
                             smoothing_num=1,
                             alpha=CU.get_alpha(23.85, 50),
                             i_func=T.i_sense_digamma_quad,
                             config=Sep19Config,
                             set_name='Sep19') for num in datnums]
            metas = dict(zip(datnums, metas))
            if datnum == 2713:
                metas[datnum].set_dat(datnum, config=Sep19Config)
                metas[datnum].row_remove = {0, 7, 10, 11, 14, 3, 1, 2}
            elif datnum == 2714:
                metas[datnum].set_dat(datnum, config=Sep19Config)
                metas[datnum].row_remove = {1, 4, 6, 15}
            elif datnum == 2715:
                metas[datnum].set_dat(datnum, config=Sep19Config)
                metas[datnum].row_remove = {3}
            elif datnum == 2716:
                metas[datnum].set_dat(datnum, config=Sep19Config)
                metas[datnum].row_remove = {18, 0, 1, 19, 6}
            elif datnum == 2717:
                metas[datnum].set_dat(datnum, config=Sep19Config)
                metas[datnum].row_remove = {0, 1, 3, 4, 9, 14, 16, 5, 6, 8, 12, 18, 19}  # Charge step at end
            elif datnum == 2718:
                metas[datnum].set_dat(datnum, config=Sep19Config)
                metas[datnum].row_remove = {0}
            elif datnum == 2719:
                metas[datnum].set_dat(datnum, config=Sep19Config)
                metas[datnum].row_remove = {10}
            elif datnum == 2720:
                metas[datnum].set_dat(datnum, config=Sep19Config)
                metas[datnum].row_remove = {7, 14}
            # elif datnum == 0:
            #     metas[datnum].set_dat(datnum, config=Sep19Config)
            #     metas[datnum].row_remove = {}
            else:  # Catch all others
                metas[datnum].set_dat(datnum, config=Sep19Config)
                metas[datnum].row_remove = {}
            meta = metas[datnum]
        else:
            raise ValueError(f'dat[{datnum}] not in set [{set_name}]')

    elif set_name.lower() == 'mar19_gamma_entropy':
        """This is for the slower entropy scans (the odd datnums), even datnums are faster transition only scans"""
        from src.Configs import Mar19Config
        if datdf is None:
            datdf = get_exp_df('mar19', dfname='Apr20')
        datnums = InDepthData.get_datnums(set_name)
        if datnum in datnums:
            # Set defaults
            metas = [DatMeta(datnum=None, datname='base', datdf=datdf,
                             # Don't want to load all dats unless necessary because it's slow
                             dt=get_mar19_dt(750 / 50 * np.sqrt(2)),  # no single DCbias dat for Mar19
                             rows=None,
                             row_remove=None,
                             bin_data_points=1,
                             i_spacing_y=-2,
                             i_spacing_x=-0.3,
                             e_spacing_y=-3,
                             e_spacing_x=0,
                             smoothing_num=1,
                             alpha=CU.get_alpha(6.727, 100),
                             i_func=T.i_sense_digamma_quad,
                             config=Mar19Config,
                             set_name='Mar19') for num in datnums]
            metas = dict(zip(datnums, metas))
            if datnum == 2689:
                metas[datnum].set_dat(datnum, config=Mar19Config)
                metas[datnum].row_remove = {7}
            elif datnum == 2691:
                metas[datnum].set_dat(datnum, config=Mar19Config)
                metas[datnum].row_remove = {5}
            elif datnum == 2693:
                metas[datnum].set_dat(datnum, config=Mar19Config)
                metas[datnum].row_remove = {0}
            elif datnum == 2695:
                metas[datnum].set_dat(datnum, config=Mar19Config)
                metas[datnum].row_remove = {0, 3, 8, 9}
            elif datnum == 2697:
                metas[datnum].set_dat(datnum, config=Mar19Config)
                metas[datnum].row_remove = {}
            elif datnum == 2699:
                metas[datnum].set_dat(datnum, config=Mar19Config)
                metas[datnum].row_remove = {}
            elif datnum == 2701:
                metas[datnum].set_dat(datnum, config=Mar19Config)
                metas[datnum].row_remove = {}
            elif datnum == 2703:
                metas[datnum].set_dat(datnum, config=Mar19Config)
                metas[datnum].row_remove = {}
            else:
                metas[datnum].set_dat(datnum, config=Mar19Config)
            meta = metas[datnum]
        else:
            raise ValueError(f'dat[{datnum}] not in set [{set_name}]')
    else:

        raise ValueError(f'Set [{set_name}] does not exist in get_dat_setup')

    return meta


class InDepthData(object):
    class Plot:
        """Neat place to put all individual plot fns"""

        @staticmethod
        def plot_avg_i(idd, ax=None, centered=True, sub_lin=False, sub_const=False, sub_quad=False):
            """
            Only avg_i data and fit (with data labelled only)

            @param centered: Move center of transition to 0,0
            @type centered: bool
            @param idd: instance of InDepthData
            @type idd: InDepthData
            @param ax: the axes to add to
            @type ax: plt.Axes
            @return: None
            @rtype: None
            """
            if ax is None:
                fig, ax = plt.subplots(1, figsize=idd.fig_size)

            x = idd.i_avg_fit.x
            y = idd.i_avg_fit.data
            y_fit = idd.i_avg_fit.best_fit

            if sub_quad is True and 'quad' in idd.i_avg_fit.params.keys():
                y = y - idd.i_avg_fit.quad * (x - idd.i_avg_fit.mid) ** 2
                y_fit = y_fit - idd.i_avg_fit.quad * (x - idd.i_avg_fit.mid) ** 2

            if sub_lin is True:
                y = y - idd.i_avg_fit.lin * (x - idd.i_avg_fit.mid)
                y_fit = y_fit - idd.i_avg_fit.lin * (x - idd.i_avg_fit.mid)

            if sub_const is True:
                y = y - idd.i_avg_fit.const
                y_fit = y_fit - idd.i_avg_fit.const

            if centered is True:
                x = x - idd.i_avg_fit.mid

            ax.scatter(x, y, s=1)
            c = ax.collections[-1].get_facecolor()
            ax.scatter([], [], s=10, c=c, label=f'[{idd.datnum}]')  # Add larger label
            ax.plot(x, y_fit, c='C3', linewidth=1)
            return x, y

        @staticmethod
        def plot_int_e(idd, ax, centered=True):
            """
            Only Integrated data of e_avg_fit labelled
            @param idd: instance of InDepthData
            @type idd: InDepthData
            @param ax: axes to add to
            @type ax: plt.Axes
            @param centered: Center plunger gate on 0
            @type centered: bool
            @return: None
            @rtype: None
            """
            if centered is True:
                x_shift = idd.i_avg_fit.mid
            else:
                x_shift = 0

            ax.scatter(idd.e_avg_fit.x - x_shift, idd.e_avg_fit.integrated, s=1)
            c = ax.collections[-1].get_facecolor()
            ax.scatter([], [], s=10, c=c, label=f'[{idd.datnum}]')  # Add larger label

    class Fitting:
        """Neat place to put all fitting procedures"""

        @staticmethod
        def i_avg_fit(idd, params=None, func=None):
            if params is None:
                params = [idd.i_params[0]]
            if func is None:
                func = idd.i_func
            params = CU.ensure_params_list(params, idd.i_avg, verbose=True)
            fit = FitData(T.transition_fits(idd.x, idd.i_avg, params=params, func=func)[0])
            idd.i_avg_fit = fit
            return fit

        @staticmethod
        def e_avg_fit(idd, params=None):
            if params is None:
                params = [idd.e_params[0]]
            dc = idd.setup_meta.dc
            dat = idd.setup_meta.dat
            params = CU.ensure_params_list(params, idd.e_avg, verbose=True)
            fit = FitData(E.entropy_fits(idd.x, idd.e_avg, params=params)[0],
                          dt=idd.setup_meta.dt,
                          amp=idd.i_avg_fit.amp
                          )
            idd.e_avg_fit = fit
            return fit

        @staticmethod
        def e_ln2_fit(idd):
            params = CU.edit_params(idd.e_avg_fit.params, 'dS', np.log(2), vary=False)
            dc = idd.setup_meta.dc
            dat = idd.setup_meta.dat
            params = CU.ensure_params_list(params, idd.e_avg, verbose=False)
            fit = FitData(E.entropy_fits(idd.x, idd.e_avg, params=params)[0],
                          dt=idd.setup_meta.dt,
                          amp=idd.i_avg_fit.amp
                          )
            idd.e_ln2_fit = fit
            return fit

        @staticmethod
        def e_avg_dt_ln2(idd):
            """Doesn't actually need to do any fitting, just copying and then changing dT in the FitData object"""
            fake_fit = copy.deepcopy(idd.e_avg_fit)
            fake_fit.dt = idd.e_avg_fit.scaling_dt * idd.e_avg_fit.integrated[-1] / np.log(2)
            fake_fit.set_sf()
            idd.e_avg_dt_ln2 = fake_fit
            return fake_fit

        @staticmethod
        def i_amp_ln2_fit(idd):
            new_amp = idd.i_avg_fit.amp * idd.e_avg_fit.integrated[-1] / np.log(2)
            params = CU.edit_params(idd.i_avg_fit.params, 'amp', new_amp, vary=False)
            for par in params:
                params[par].vary = False
            params = CU.ensure_params_list(params, idd.i_avg, verbose=False)
            fit = FitData(T.transition_fits(idd.x, idd.i_avg, params=params, func=idd.i_func)[0])
            idd.i_amp_ln2_fit = fit
            return fit

    class FullPlots:
        @staticmethod
        def plot_i_sense_by_row(idd, raw=False, smoothed=True, show_tables=False, sub_poly=True):
            meta = idd.setup_meta
            if raw is True:
                fig, axs = PF.make_axes(1, single_fig_size=idd.fig_size)
                ax = axs[0]
                x, y = idd.x, idd.y_isense
                if sub_poly is True:
                    x, y = sub_poly_from_data(x, y, idd.i_fits)
                PF.waterfall_plot(x, y, ax=ax, y_spacing=meta.i_spacing_y, x_spacing=meta.i_spacing_x,
                                  every_nth=1,
                                  plot_args={'s': 1},
                                  ptype='scatter', label=True, cmap_name=idd.cmap_name, index=meta.rows)
                PF.ax_setup(ax, f'I_sense data for dat[{meta.dat.datnum}]', meta.dat.Logs.x_label, 'I_sense /nA',
                            legend=True)
                PF.add_standard_fig_info(fig)

            if smoothed is True:
                fig, axs = PF.make_axes(1, single_fig_size=idd.fig_size)
                ax = axs[0]
                if meta.smoothing_num > 1:
                    ysmooth = savgol_filter(idd.y_isense, meta.smoothing_num, 1)
                else:
                    ysmooth = idd.y_isense
                xi = (CU.get_data_index(idd.x, idd.i_avg_fit.mid - idd.view_width),
                      CU.get_data_index(idd.x, idd.i_avg_fit.mid + idd.view_width))
                x, y = idd.x[xi[0]:xi[1]], ysmooth[:, xi[0]:xi[1]]
                if sub_poly is True:
                    x, y = sub_poly_from_data(x, y, idd.i_fits)
                y_add, x_add = PF.waterfall_plot(x, y, ax=ax,
                                                 y_spacing=meta.i_spacing_y,
                                                 x_spacing=meta.i_spacing_x,
                                                 every_nth=1, plot_args={'s': 1}, ptype='scatter', label=True,
                                                 cmap_name=idd.cmap_name, index=meta.rows)
                y_fits = np.array([fit.eval(x=idd.x[xi[0]:xi[1]]) for fit in idd.i_fits])
                x, y_fits = idd.x[xi[0]:xi[1]], y_fits
                if sub_poly is True:
                    x, y_fits = sub_poly_from_data(x, y_fits, idd.i_fits)
                PF.waterfall_plot(x, y_fits, ax=ax, y_add=y_add, x_add=x_add, color='C3', ptype='plot')
                PF.ax_setup(ax, f'Smoothed I_sense data for dat[{meta.dat.datnum}]\nwith fits', meta.dat.Logs.x_label,
                            'I_sense /nA',
                            legend=True)
                PF.add_standard_fig_info(fig)

                if show_tables is True:
                    df = CU.fit_info_to_df(idd.i_fits, uncertainties=idd.uncertainties, sf=3, index=meta.rows)
                    PF.plot_df_table(df, title=f'I_sense_fit info for dat[{meta.dat.datnum}]')

        @staticmethod
        def plot_average_i_sense(idd, avg=True, others=False, show_tables=False, sub_poly=True):
            meta = idd.setup_meta
            if avg is True:
                fig, axs = PF.make_axes(1, single_fig_size=idd.fig_size)

                ax = axs[0]
                # PF.display_1d(self.x, self.i_avg, ax, scatter=True, label='Averaged data')
                xi = (
                    CU.get_data_index(idd.x, idd.i_avg_fit.mid - idd.view_width),
                    CU.get_data_index(idd.x, idd.i_avg_fit.mid + idd.view_width))
                x, y = idd.x[xi[0]:xi[1]], idd.i_avg[xi[0]:xi[1]]
                if sub_poly is True:
                    x, y = sub_poly_from_data(x, y, idd.i_avg_fit.fit)
                PF.display_1d(x, y, ax, scatter=True, label='self.i_avg')
                # ax.plot(self.x_i_fit_avg, i_fit_avg.best_fit, c='C3', label='Best fit')

                x, y_fit = idd.i_avg_fit.x, idd.i_avg_fit.best_fit
                if sub_poly is True:
                    x, y_fit = sub_poly_from_data(x, y_fit, idd.i_avg_fit.fit)
                ax.plot(x, y_fit, c='C3', label='i_avg_fit.best_fit')
                PF.ax_setup(ax, f'Dat[{meta.dat.datnum}]:Averaged data with fit', meta.dat.Logs.x_label, 'I_sense /nA',
                            legend=True)

                if show_tables is True:
                    df = CU.fit_info_to_df([idd.i_avg_fit.fit], uncertainties=idd.uncertainties, sf=3,
                                           index=meta.rows)
                    df.pop('index')
                    PF.plot_df_table(df, title=f'Dat[{meta.dat.datnum}]:I_sense fit values no additional forcing')

                PF.add_standard_fig_info(fig)

            if others is True:
                fig, axs = PF.make_axes(2, single_fig_size=idd.fig_size)
                ax = axs[0]
                # PF.display_1d(self.x, self.i_avg, ax, scatter=True, label='Averaged data')
                PF.display_1d(idd.x, idd.i_avg, ax, scatter=True, label='self.i_avg')
                # ax.plot(x_i_fit_avg, i_fit_ln2.best_fit, c='C3', label='Ln(2) amplitude fit')
                ax.plot(idd.i_amp_ln2_fit.x, idd.i_amp_ln2_fit.best_fit, c='C3', label='i_amp_ln2_fit.best_fit')
                PF.ax_setup(ax,
                            f'Dat[{meta.dat.datnum}]:Averaged I_sense data with\nwith amp forced s.t. int_dS = Ln(2)',
                            meta.dat.Logs.x_label, 'I_sense /nA', legend=True)
                PF.add_standard_fig_info(fig)

                if show_tables is True:
                    df = CU.fit_info_to_df([idd.i_amp_ln2_fit.fit], uncertainties=idd.uncertainties, sf=3,
                                           index=meta.rows)
                    df.pop('index')
                    PF.plot_df_table(df,
                                     title=f'Dat[{meta.dat.datnum}]:I_sense fit values with amp forced s.t. int_dS = Ln(2)')

        @staticmethod
        def plot_entropy_by_row(idd, raw=False, smoothed=True, show_tables=False):
            meta = idd.setup_meta

            if raw is True:
                fig, axs = PF.make_axes(1, single_fig_size=idd.fig_size)
                ax = axs[0]
                y_add, x_add = PF.waterfall_plot(idd.x, idd.y_entr, ax=ax, y_spacing=meta.e_spacing_y,
                                                 x_spacing=meta.e_spacing_x,
                                                 every_nth=1, plot_args={'s': 1}, ptype='scatter', label=True,
                                                 cmap_name=idd.cmap_name, index=meta.rows)
                PF.ax_setup(ax, f'Entropy_r data for dat[{meta.dat.datnum}]', meta.dat.Logs.x_label, 'Entr /nA',
                            legend=True)
                PF.add_standard_fig_info(fig)

            if smoothed is True:
                fig, axs = PF.make_axes(1, single_fig_size=idd.fig_size)
                ax = axs[0]
                if meta.smoothing_num > 1:
                    ysmooth = savgol_filter(idd.y_entr, meta.smoothing_num, 1)
                else:
                    ysmooth = idd.y_entr
                xi = (CU.get_data_index(idd.x, idd.i_avg_fit.mid - idd.view_width),
                      CU.get_data_index(idd.x, idd.i_avg_fit.mid + idd.view_width))
                y_add, x_add = PF.waterfall_plot(idd.x[xi[0]:xi[1]], ysmooth[:, xi[0]:xi[1]], ax=ax,
                                                 y_spacing=meta.e_spacing_y,
                                                 x_spacing=meta.e_spacing_x,
                                                 every_nth=1,
                                                 plot_args={'s': 1}, ptype='scatter', label=True,
                                                 cmap_name=idd.cmap_name,
                                                 index=meta.rows)
                y_fits = np.array([fit.eval(x=idd.x[xi[0]:xi[1]]) for fit in idd.e_fits])
                PF.waterfall_plot(idd.x[xi[0]:xi[1]], y_fits, ax=ax, y_add=y_add, x_add=x_add, color='C3',
                                  ptype='plot')
                PF.ax_setup(ax, f'Dat[{meta.dat.datnum}]:Smoothed entropy_r data\nwith fits', meta.dat.Logs.x_label,
                            'Entr /nA', legend=True)
                PF.add_standard_fig_info(fig)

                if show_tables is True:
                    df = CU.fit_info_to_df(idd.e_fits, uncertainties=idd.uncertainties, sf=3, index=meta.rows)
                    PF.plot_df_table(df, title=f'Entropy_R_fit info for dat[{meta.dat.datnum}]')
                PF.add_standard_fig_info(fig)

        @staticmethod
        def plot_average_entropy(idd, avg=True, others=False, show_tables=False):
            meta = idd.setup_meta
            if avg is True:
                fig, axs = PF.make_axes(1, single_fig_size=idd.fig_size)
                ax = axs[0]
                # PF.display_1d(self.x, e_y_avg, ax, scatter=True, label='Averaged data')
                PF.display_1d(idd.x, idd.e_avg, ax, scatter=True, label='e_avg')
                # ax.plot(x_e_fit_avg, e_fit_avg.best_fit, c='C3', label='Best fit')
                ax.plot(idd.e_avg_fit.x, idd.e_avg_fit.best_fit, c='C3', label='e_avg_fit.best_fit')
                PF.ax_text(ax, f'dT={idd.e_avg_fit.scaling_dt * meta.alpha / Const.kb * 1000:.3f}mK', loc=(0.02, 0.6))
                PF.ax_setup(ax, f'Dat[{meta.dat.datnum}]:Averaged Entropy R data with fit', meta.dat.Logs.x_label,
                            'Entropy R /nA',
                            legend=True)

                if show_tables is True:
                    df = CU.fit_info_to_df([idd.e_avg_fit.fit], uncertainties=idd.uncertainties, sf=3,
                                           index=meta.rows)
                    df.pop('index')
                    PF.plot_df_table(df,
                                     title=f'Dat[{meta.dat.datnum}]:Entropy R fit values with no additional forcing')

                PF.add_standard_fig_info(fig)

            if others is True:
                fig, axs = PF.make_axes(2, single_fig_size=idd.fig_size)
                # region Forced to dS = Ln2
                ax = axs[0]
                # PF.display_1d(self.x, e_y_avg, ax, scatter=True, label='Averaged data')
                PF.display_1d(idd.x, idd.e_avg, ax, scatter=True, label='e_avg')
                # ax.plot(x_e_fit_avg, e_fit_ln2.best_fit, c='C3', label='Ln(2) fit')
                ax.plot(idd.e_ln2_fit.x, idd.e_ln2_fit.best_fit, c='C3', label='e_ln2_fit.best_fit')
                PF.ax_text(ax, f'dT={idd.e_ln2_fit.scaling_dt * meta.alpha / Const.kb * 1000:.3f}mK', loc=(0.02, 0.6))
                PF.ax_setup(ax, f'Dat[{meta.dat.datnum}]:Averaged Entropy R data with Ln(2) fit', meta.dat.Logs.x_label,
                            'Entropy R /nA',
                            legend=True)
                PF.add_standard_fig_info(fig)

                if show_tables is True:
                    df = CU.fit_info_to_df([idd.e_ln2_fit.fit], uncertainties=idd.uncertainties, sf=3,
                                           index=meta.rows)
                    df.pop('index')
                    PF.plot_df_table(df, title=f'Dat[{meta.dat.datnum}]:Entropy R fit values with dS forced to Ln2')

                # region Forced dT s.t. int_data dS = ln2
                ax = axs[1]
                # PF.display_1d(self.x, e_y_avg, ax, scatter=True, label='Averaged data')
                PF.display_1d(idd.x, idd.e_avg, ax, scatter=True, label='e_avg')
                # ax.plot(x_e_fit_avg, e_fit_dt_ln2.best_fit, c='C3', label='dT forced fit')
                ax.plot(idd.e_avg_dt_ln2.x, idd.e_avg_dt_ln2.best_fit, c='C3', label='e_avg_dt_ln2.best_fit')
                PF.ax_text(ax, f'dT={idd.e_avg_dt_ln2.scaling_dt * meta.alpha / Const.kb * 1000:.3f}mK',
                           loc=(0.02, 0.6))
                PF.ax_setup(ax,
                            f'Dat[{meta.dat.datnum}]:Averaged Entropy R data\nwith dT forced s.t. int_data dS = Ln2',
                            meta.dat.Logs.x_label, 'Entropy R /nA',
                            legend=True)
                PF.add_standard_fig_info(fig)

                if show_tables is True:
                    df = CU.fit_info_to_df([idd.e_avg_dt_ln2.fit], uncertainties=idd.uncertainties, sf=3,
                                           index=meta.rows)
                    df.pop('index')
                    PF.plot_df_table(df,
                                     title=f'Dat[{meta.dat.datnum}]:Entropy R fit values\nwith dT forced s.t. int_data dS = Ln2')
                    # endregion
                    PF.add_standard_fig_info(fig)

        @staticmethod
        def plot_integrated(idd, avg=True, others=False):
            meta = idd.setup_meta
            if avg is True:
                # region dT from DCbias, amp from I_sense, also int of e_fit_avg
                fig, axs = PF.make_axes(1, single_fig_size=idd.fig_size)
                ax = axs[0]
                # PF.display_1d(self.x, int_avg, ax, label='Averaged data')
                PF.display_1d(idd.e_avg_fit.x, idd.e_avg_fit.integrated, ax, label='e_avg_int')
                # ax.plot(x_e_fit_avg, int_of_fit, c='C3', label='integrated best fit')
                ax.plot(idd.e_avg_fit.x, idd.e_avg_fit.fit_integrated, c='C3', label='int_of_fit')
                PF.ax_setup(ax, f'Dat[{meta.dat.datnum}]:Integrated Entropy\ndT from DCbias for data and fit',
                            meta.dat.Logs.x_label,
                            'Entropy /kB')
                _add_ln3_ln2(ax)
                _add_peak_final_text(ax, idd.e_avg_fit.integrated, idd.e_avg_fit.fit_integrated)
                ax.legend(loc='lower right')
                PF.ax_text(ax, f'dT = {idd.e_avg_fit.scaling_dt * meta.alpha / Const.kb * 1000:.3f}mK\n'
                               f'amp = {idd.i_avg_fit.amp:.3f}nA\n'
                               f'int_avg dS={idd.e_avg_fit.integrated[-1] / np.log(2):.3f}kBLn2\n'
                               f'int_of_fit dS={idd.e_avg_fit.fit_integrated[-1] / np.log(2):.3f}kBLn2',
                           loc=(0.02, 0.7), fontsize=8)

                PF.add_standard_fig_info(fig)
                # endregion
            if others is True:
                fig, axs = PF.make_axes(3, single_fig_size=idd.fig_size)
                # region dT adjusted s.t. integrated_data has dS = ln2, fit with that dt forced then integrated
                ax = axs[0]
                # PF.display_1d(self.x, int_avg_dt_ln2, ax, label='Averaged data')
                PF.display_1d(idd.e_avg_dt_ln2.x, idd.e_avg_dt_ln2.integrated, ax, label='e_avg_dt_ln2')
                # ax.plot(x_e_fit_avg, int_of_fit_dt_ln2, c='C3', label='integrated fit\nwith dT forced')
                ax.plot(idd.e_avg_dt_ln2.x, idd.e_avg_dt_ln2.fit_integrated, c='C3', label='fit')
                PF.ax_setup(ax, f'Dat[{meta.dat.datnum}]:Integrated Entropy\ndT forced s.t. int_ds=Ln2',
                            meta.dat.Logs.x_label,
                            'Entropy /kB')
                _add_ln3_ln2(ax)
                _add_peak_final_text(ax, idd.e_avg_dt_ln2.integrated, idd.e_avg_dt_ln2.fit_integrated)
                ax.legend(loc='lower right')
                PF.ax_text(ax, f'dT of forced fit={idd.e_avg_dt_ln2.scaling_dt * meta.alpha / Const.kb * 1000:.3f}mK\n'
                               f'amp = {idd.i_avg_fit.amp:.3f}nA\n'
                               f'int_avg_dt_ln2 dS={idd.e_avg_dt_ln2.integrated[-1] / np.log(2):.3f}kBLn2\n'
                               f'int_fit_dt_ln2 dS={idd.e_avg_dt_ln2.fit_integrated[-1] / np.log(2):.3f}kBLn2',
                           loc=(0.02, 0.7), fontsize=8)

    @staticmethod
    def get_default_plot_list():
        show_plots = {
            'i_sense': False,  # waterfall w/fit
            'i_sense_raw': False,  # waterfall raw
            'i_sense_avg': False,  # averaged_i_sense
            'i_sense_avg_others': False,  # forced fits
            'entr': False,  # waterfall w/fit
            'entr_raw': False,  # waterfall raw
            'avg_entr': False,  # averaged_entr
            'avg_entr_others': False,  # forced fits
            'int_ent': False,  # integrated_entr
            'int_ent_others': False,  # forced fits
            'tables': False  # Fit info tables
        }
        return show_plots

    @staticmethod
    def get_datnums(set_name):
        """
        Gets the list of datnums being used in named dataset
        @param set_name: Name of dataset (same as for InDepthData)
        @type set_name: str
        @return: list of datnums
        @rtype: list[int]
        """
        if set_name.lower() == 'jan20':
            datnums = [1533, 1501]
        elif set_name.lower() == 'jan20_gamma':
            datnums = [1492, 1495, 1498, 1501, 1504, 1507, 1510, 1513, 1516, 1519, 1522, 1525, 1528]
        elif set_name.lower() == 'jan20_gamma_2':
            datnums = [1533, 1536, 1539, 1542, 1545, 1548, 1551, 1554, 1557, 1560, 1563, 1566]
        elif set_name.lower() == 'sep19_gamma':
            datnums = list(set(range(2713, 2729 + 1)) - {2721, 2722})
        elif set_name.lower() == 'mar19_gamma_entropy':
            datnums = list(range(2689, 2711 + 1, 2))
        else:
            logger.warning(f'[InDepthData.get_datnums]: There are no datnums for set [{set_name}]')
            datnums = None
        return datnums

    def __init__(self, datnum, plots_to_show=None, set_name='Jan20_gamma', run_fits=True, show_plots=True, datdf=None):
        # region Data Setup
        self.datnum = datnum
        self.set_name = set_name
        self.setup_meta = get_dat_setup(datnum, set_name=set_name, datdf=datdf)
        # endregion

        # region Data
        self.x = None
        self.dx = None
        self.y_isense = None
        self.y_entr = None
        self.set_data()  # Sets the above attributes using info from setup_meta and setup_meta.dat
        # endregion

        # region Row fits init
        self.i_func = self.setup_meta.i_func if self.setup_meta.i_func is not None else T.i_sense
        self.i_params = None
        self.i_fits = None

        self.e_params = None
        self.e_fits = None
        # endregion

        # region Average Data init
        self.i_avg = None
        self.e_avg = None
        # endregion

        # region Fit Data init
        self.i_avg_fit = None
        self.i_amp_ln2_fit = None  # amp s.t. int_data dS = Ln2

        self.e_avg_fit = None  # Regular
        self.e_ln2_fit = None
        self.e_avg_dt_ln2 = None  # scaling dT s.t. int_data dS = Ln2
        # endregion

        if run_fits is True:
            self.run_all_fits(i_params=self.i_params, i_func=self.i_func, e_params=self.e_params)  # params can be None

        # region Plot related
        if plots_to_show is None:
            plots_to_show = InDepthData.get_default_plot_list()
        self.plots_to_show = plots_to_show
        self.view_width = 3000
        self.fig_size = (5, 5)
        self.cmap_name = 'tab10'
        # endregion

        # region Table related
        self.uncertainties = True  # Whether to show uncertainties in tables or not
        # endregion

        if show_plots is True and run_fits is True:
            self.plot_all_plots()

    def set_data(self):
        dat = self.setup_meta.dat
        self.x = CU.bin_data(dat.Data.x_array, self.setup_meta.bin_data_points)
        self.dx = np.abs(self.x[-1] - self.x[0]) / len(self.x)
        rows = self.setup_meta.rows
        if rows is None:
            print(f'For dat[{self.datnum}] - Loading all data rows')
            self.y_isense = CU.bin_data(dat.Data.i_sense, self.setup_meta.bin_data_points)
            self.y_entr = CU.bin_data(dat.Entropy.entr, self.setup_meta.bin_data_points)
        else:
            print(f'For dat[{self.datnum}] - Loading data rows: {rows}')
            self.y_isense = np.array([CU.bin_data(dat.Data.i_sense[i], self.setup_meta.bin_data_points) for i in rows])
            self.y_entr = np.array([CU.bin_data(dat.Entropy.entr[i], self.setup_meta.bin_data_points) for i in rows])

    def plot_all_plots(self):
        show_plots = self.plots_to_show
        self.FullPlots.plot_i_sense_by_row(self, raw=show_plots['i_sense_raw'], smoothed=show_plots['i_sense'],
                                           show_tables=show_plots['tables'])
        self.FullPlots.plot_entropy_by_row(self, raw=show_plots['entr_raw'], smoothed=show_plots['entr'],
                                           show_tables=show_plots['tables'])
        self.FullPlots.plot_average_i_sense(self, avg=show_plots['i_sense_avg'],
                                            others=show_plots['i_sense_avg_others'],
                                            show_tables=show_plots['tables'])
        self.FullPlots.plot_average_entropy(self, avg=show_plots['avg_entr'], others=show_plots['i_sense_avg_others'],
                                            show_tables=show_plots['tables'])
        self.FullPlots.plot_integrated(self, avg=show_plots['int_ent'], others=show_plots['int_ent_others'])

    def run_all_fits(self, i_params=None, i_func=None, e_params=None):
        self.i_params, self.i_func, self.i_fits = self.fit_isenses(params=i_params, func=i_func)
        self.e_params, self.e_fits = self.fit_entropys(params=e_params)

        self.make_averages()

        self.i_avg_fit = self.Fitting.i_avg_fit(self, params=i_params, func=i_func)

        self.e_avg_fit = self.Fitting.e_avg_fit(self, params=e_params)
        self.e_ln2_fit = self.Fitting.e_ln2_fit(self, )
        self.e_avg_dt_ln2 = self.Fitting.e_avg_dt_ln2(
            self, )  # Not really fitting, just setting dT for scaling differently

        self.i_amp_ln2_fit = self.Fitting.i_amp_ln2_fit(self, )  # Has to go after e_avg_fit

    def make_averages(self):
        self.i_avg, _ = np.array(CU.average_data(self.y_isense,
                                                 [CU.get_data_index(self.x, fit.best_values['mid'])
                                                  for fit in self.i_fits]))
        self.e_avg, _ = np.array(CU.average_data(self.y_entr,
                                                 [CU.get_data_index(self.x, fit.best_values['mid'])
                                                  for fit in self.i_fits]))

    def fit_isenses(self, params=None, func=None):
        x = self.x
        data = self.y_isense

        if func is None:
            if self.i_func is not None:
                func = self.i_func
            else:
                func = T.i_sense
        if params is None:
            if self.i_params is not None:
                params = self.i_params
            else:
                params = T.get_param_estimates(x, data)
                if func == T.i_sense_digamma:
                    for par in params:
                        T._append_param_estimate_1d(par, ['g'])
                elif func == T.i_sense_digamma_quad:
                    for par in params:
                        T._append_param_estimate_1d(par, ['g', 'quad'])
        params = CU.ensure_params_list(params, data, verbose=True)
        fits = T.transition_fits(x, data, params=params, func=func)
        self.i_params = params
        self.i_func = func
        self.i_fits = fits
        return params, func, fits

    def fit_entropys(self, params=None):
        x = self.x
        data = self.y_entr
        mids = [fit.best_values['mid'] for fit in self.i_fits]
        thetas = [fit.best_values['theta'] for fit in self.i_fits]
        if params is None:
            if self.e_params is not None:
                params = self.e_params
            else:
                params = E.get_param_estimates(x, data, mids, thetas)
        params = CU.ensure_params_list(params, data, verbose=True)
        fits = E.entropy_fits(x, data, params=params)
        self.e_params = params
        self.e_fits = fits
        return params, fits


def get_exp_df(exp_name='mar19', dfname='Apr20'):
    datdf: DF.DatDF
    if exp_name.lower() == 'mar19':
        from src.Configs import Mar19Config
        mar19_config_switcher = CU.switch_config_decorator_maker(Mar19Config)
        datdf = CU.wrapped_call(mar19_config_switcher, (lambda: DF.DatDF(dfname=dfname)))
        setupdf = CU.wrapped_call(mar19_config_switcher, (lambda: SF.SetupDF()))
        return datdf, setupdf
    elif exp_name.lower() == 'sep19':
        from src.Configs import Sep19Config
        sep19_config_switcher = CU.switch_config_decorator_maker(Sep19Config)
        datdf = CU.wrapped_call(sep19_config_switcher, (lambda: DF.DatDF(dfname=dfname)))
        setupdf = CU.wrapped_call(sep19_config_switcher, (lambda: SF.SetupDF()))
        return datdf, setupdf
    elif exp_name.lower() == 'jan20':
        from src.Configs import Jan20Config
        jan20_config_switcher = CU.switch_config_decorator_maker(Jan20Config)
        datdf = CU.wrapped_call(jan20_config_switcher, (lambda: DF.DatDF(dfname=dfname)))
        setupdf = CU.wrapped_call(jan20_config_switcher, (lambda: SF.SetupDF()))
        return datdf, setupdf
    elif exp_name.lower() == 'jun20':
        from src.Configs import Jun20Config
        jun20_config_switcher = CU.switch_config_decorator_maker(Jun20Config)
        datdf = CU.wrapped_call(jun20_config_switcher, (lambda: DF.DatDF(dfname=dfname)))
        setupdf = CU.wrapped_call(jun20_config_switcher, (lambda: SF.SetupDF()))
        return datdf, setupdf
    else:
        raise ValueError(f'exp_name must be in ["mar19", "sep19"]. [{exp_name}] was not found.')


def get_mar19_dt(i_heat):
    mar19_dcdats = list(range(3385, 3410 + 1, 2))  # 0.8mV steps. Repeat scan of theta at DC bias steps
    # mar19_dcdats = list(range(3386, 3410 + 1, 2))  # 3.1mV steps. Repeat scan of theta at DC bias steps
    from src.Configs import Mar19Config
    datdf = get_exp_df('mar19')
    mdcs = [C.DatHandler.get_dat(num, 'base', datdf, config=Mar19Config) for num in mar19_dcdats]
    m_xs = [dat.Logs.dacs[0] / 10 for dat in mdcs]
    m_ys = [dat.Transition.theta for dat in mdcs]
    quad = lm.models.QuadraticModel()
    m_result = quad.fit(m_ys, x=m_xs)
    mx = np.average([m_result.eval(x=i_heat), m_result.eval(x=-i_heat)])
    mn = np.nanmin(m_result.eval(x=np.linspace(-5, 5, 1000)))
    dt = (mx - mn) / 2
    return dt


def make_row_ranges(dat_idd, set_name=None, chunks=5, remove_rows=True):
    """
    Makes a set of IDDs with row range st a single scan is broken in to #chunks
    @param remove_rows: Whether to remove the rows previously marked in datmeta or not (will not affect row ranges)
    @type remove_rows: bool
    @param dat_idd: dat or IDD instance
    @type dat_idd: Union[D.Dat, InDepthData]
    @param set_name: Set_name in get_dat_setup
    @type set_name: str
    @param chunks: How many chunks to split data into
    @type chunks: int
    @return: list of IDD instances with incrementing chunks of rows (excluding any rows in row_remove in the setup_meta)
    @rtype: List[InDepthData]
    """
    if isinstance(dat_idd, InDepthData):
        idd = dat_idd
        dat = idd.setup_meta.dat
        if set_name is None:
            set_name = dat_idd.setup_meta.set_name
    else:
        assert isinstance(dat_idd, D.Dat)
        assert set_name is not None
        dat = dat_idd
        idd = InDepthData(dat.datnum, set_name=set_name)

    total_rows = dat.Logs.y_array.shape[0]
    starts = np.round(np.arange(0, total_rows, total_rows / chunks)).astype(int)
    ends = np.append(starts[1:], total_rows).astype(int)
    IDD_ranges = []
    for start, end in zip(starts, ends):
        rows = list(range(start, end, 1))
        new_idd = InDepthData(dat.datnum, None, set_name, False, False)
        new_idd.setup_meta.rows = rows
        if remove_rows is True:
            new_idd.setup_meta.set_rows()  # remove any rows in row_remove
        else:
            new_idd.setup_meta.row_remove = {}
        new_idd.set_data()
        new_idd.run_all_fits()
        IDD_ranges.append(new_idd)
    return IDD_ranges


class CompareResult(object):
    def __init__(self):
        self.i_df = pd.DataFrame()  # fit values
        self.i_dfu = pd.DataFrame()  # fit uncertainties
        self.i_df_text = pd.DataFrame()  # text fit value +- uncertainty

        self.e_df = pd.DataFrame()
        self.e_dfu = pd.DataFrame()
        self.e_df_text = pd.DataFrame()


def get_additional_columns(idds):
    """
    When passed a list of IDD instances it will return lists of differences to be used for a df table for example
    @param idds:  list of IDDs to get additional column names/values for
    @type idds: List[InDepthData]
    @return: list of col_names, and col_values
    @rtype: (List[str], List[list])
    """
    datnums = [idd.datnum for idd in idds]
    set_names = [idd.setup_meta.set_name for idd in idds]
    rows = [idd.setup_meta.rows for idd in idds]
    row_starts = [row[0] for row in rows]
    row_ends = [row[-1] for row in rows]
    col_names = []
    col_values = []
    if len(set(set_names)) != 1:  # then some are from different sets
        col_names.append('Set')
        col_values.append([s for s in set_names])
    if len(set(datnums)) != 1:  # then some are different datnums
        col_names.append('Datnum')
        col_values.append([num for num in datnums])
    if len(set(row_starts)) != 1 or len(set(row_ends)) != 1:
        col_names.append('Rows')
        col_values.append([f'{rs}:{re}' for rs, re in zip(row_starts, row_ends)])
    return col_names, col_values


def compare_IDDs(IDDs, pre_title='Data Comparison', auto_plot=False, fits=('i', 'e')):
    """
        Compare fit params between IDD's passed in.
        @param auto_plot: Whether to automatically plot tables and figures for comparison
        @type auto_plot: bool
        @param fits: which fits to compare (i for i_sense, e for entropy)
        @type fits: Union[List[str], Tuple[str]]
        @param pre_title: Description before colon on auto generated plots
        @type pre_title: str
        @param IDDs: list of IDDs
        @type IDDs: List[InDepthData]
        @return: CompareResult object which contains dfs for fit values, uncertainties and text w/ both
        @rtype: CompareResult
        """

    res = CompareResult()

    for fit in fits:
        fit_getter = lambda idd: getattr(getattr(idd, f'{fit}_avg_fit'), 'fit')
        fits = [fit_getter(idd) for idd in IDDs]
        col_names, col_values = get_additional_columns(IDDs)
        for uncertainty, res_attr in zip([0, 1, 2], ['_df', '_df_text', '_dfu']):
            df = CU.fit_info_to_df(fits, uncertainties=uncertainty, sf=3)
            for name, vals in zip(col_names, col_values):
                df[name] = vals
            setattr(res, f'{fit}{res_attr}', df)

        if auto_plot is True:
            if fit == 'i':
                fit_name = 'I_sense'
            elif fit == 'e':
                fit_name = 'Entropy'
            else:
                raise NotImplementedError
            PF.plot_df_table(getattr(res, f'{fit}_df_text'),
                             f'{pre_title}: {fit_name} fit values with varying {col_names}')

            df = getattr(res, f'{fit}_df')
            dfu = getattr(res, f'{fit}_dfu')
            cols = [col for col in df.columns if col not in ['reduced_chi_sq', 'index'] + col_names]
            fig, axs = PF.make_axes(len(cols), single_fig_size=(3, 3))
            fig.suptitle(f'{pre_title}: {fit_name}')
            for col, ax in zip(cols, axs):
                ax.errorbar(x=df['index'], y=df[col], yerr=dfu[col], ecolor='red')
                PF.ax_setup(ax, f'{col} per row set w/ fit uncertainties', 'Index', 'Fit Value', fs=8)
            PF.add_standard_fig_info(fig)
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    return res


def _edit_fits(IDDs):
    """temporary func"""
    theta_vary = False
    g_vary = True
    for idd in IDDs:
        i_pars = idd.i_avg_fit.params
        if i_pars['theta'].vary != theta_vary or i_pars['g'].vary != g_vary:
            i_pars = CU.edit_params(i_pars, ['theta', 'g'], [27.6, 0], [False, True])
            idd.run_all_fits(i_params=i_pars)


class FitData(object):
    """
    Object for storing info about a particular 1D fit with some extra stuff (i.e. x, data, dt used, amp used)
    """

    def __getattr__(self, item):
        """Overrides behaviour when attribute is not found for Data"""
        if item.startswith('__'):  # So don't complain about things like __len__
            return super().__getattr__(self, item)
        else:
            if item in self.fit.best_values.keys():
                return self.fit.best_values[item]
            else:
                print(f'Fit parameter "{item}" does not exist for this fit\n'
                      f'Fit values are {self.fit.best_values}')
                return None

    def __init__(self, fit, dt=None, amp=None, sf=None):
        assert isinstance(fit, lm.model.ModelResult)
        self.fit = fit
        self.scaling_dt = dt
        self.scaling_amp = amp
        self.sf = sf

    @property
    def best_fit(self):
        return self.fit.best_fit

    @property
    def params(self):
        return self.fit.params

    @property
    def x(self):
        return self.fit.userkws['x']

    @property
    def data(self):
        return self.fit.data

    @property
    def dx(self):
        x = self.x
        return np.abs(x[-1] - x[0]) / len(x)

    @property
    def sf(self):
        sf = None
        if None not in [self.scaling_dt, self.scaling_amp, self.dx]:
            sf = E.scaling(self.scaling_dt, self.scaling_amp, self.dx)

        if self._sf is not None and self._sf == sf:
            return sf
        elif self._sf is not None and self._sf != sf:
            print(
                f'WARNING[FitData]: sf from dt, amp, dx = [{CU.sig_fig(sf, 3)}], stored sf is [{CU.sig_fig(self._sf)}]'
                f'\nreturned stored sf')
            return self._sf
        elif self._sf is None and sf is not None:
            self._sf = sf
            return sf
        else:
            print(f'WARNING[FitData]: sf not initialized')
            return None

    @sf.setter
    def sf(self, value):
        self._sf = value

    @property
    def integrated(self):
        return np.nancumsum(self.data) * self.sf

    @property
    def fit_integrated(self):
        return np.nancumsum(self.best_fit) * self.sf

    def set_sf(self):
        self.sf = E.scaling(self.scaling_dt, self.scaling_amp, self.dx)

    def get_integrated_with(self, dt=None, amp=None, sf=None):
        """
        Temporarily change any of the sf parameters and get the integrated result
        @type dt: float
        @type amp: float
        @type sf: float
        @return: integrated data
        @rtype: np.ndarray
        """

        if sf is None:
            if dt is None:
                dt = self.scaling_dt
            if amp is None:
                amp = self.scaling_amp
            if None in [dt, amp]:
                raise ValueError(f'dt={dt}, amp={amp}. Both need to be valid at this point...')
            sf = E.scaling(dt, amp, self.dx)
        return np.nancumsum(self.data) * sf


def _add_ln3_ln2(ax: plt.Axes):
    ax.axhline(np.log(2), linestyle=':', color='grey')
    ax.axhline(np.log(3), linestyle=':', color='grey')


def _add_peak_final_text(ax, data, fit):
    PF.ax_text(ax, f'Peak/Final /(Ln3/Ln2)\ndata, fit={(np.nanmax(data) / data[-1]) / (np.log(3) / np.log(2)):.2f}, '
                   f'{(np.nanmax(fit) / fit[-1]) / (np.log(3) / np.log(2)):.2f}',
               loc=(0.6, 0.8), fontsize=8)
