from src.Scripts.StandardImports import *
import src.DatCode.Entropy as E
import src.DatCode.Transition as T
from scipy.signal import savgol_filter
import copy
import lmfit as lm
import os
import scipy.io as sio


def _recalculate_dats(datdf: DF.DatDF, datnums: list, datname='base', dattypes: set = None, setupdf=None, config=None,
                      transition_func=None, save=True):
    """
    Just a quick fn to recalculate and save all dats to given datDF
    """
    # if datdf.config_name != cfg.current_config.__name__.split('.')[-1]:
    #     print('WARNING[_recalculate_given_dats]: Need to change config while running this. No dats changed')
    #     return
    if transition_func is not None:
        dattypes = CU.ensure_set(dattypes)
        dattypes.add(
            'suppress_auto_calculate')  # So don't do pointless calculation of transition when first initializing
    for datnum in datnums:
        dat = make_dat_standard(datnum, datname=datname, datdf=datdf, dfoption='overwrite', dattypes=dattypes,
                                setupdf=setupdf, config=config)
        if transition_func is not None:
            dat._reset_transition(fit_function=transition_func)
            if 'entropy' in dat.dattype:
                dat._reset_entropy()
        datdf.update_dat(dat, yes_to_all=True)
    if save is True:
        datdf.save()


def _add_ln3_ln2(ax: plt.Axes):
    ax.axhline(np.log(2), linestyle=':', color='grey')
    ax.axhline(np.log(3), linestyle=':', color='grey')


def _add_peak_final_text(ax, data, fit):
    PF.ax_text(ax, f'Peak/Final /(Ln3/Ln2)\ndata, fit={(np.nanmax(data) / data[-1]) / (np.log(3) / np.log(2)):.2f}, '
                   f'{(np.nanmax(fit) / fit[-1]) / (np.log(3) / np.log(2)):.2f}',
               loc=(0.6, 0.8), fontsize=8)


class DatMeta(object):
    """
    Object to store info about how to use data from given dataset
    """

    def __init__(self, datnum=None, datname=None,
                 dc_num=None, dc_name=None,
                 datdf=None,
                 dt=None,
                 rows=None, row_remove=None,
                 thin_data_points=1,
                 i_spacing_y=-1, i_spacing_x=0,
                 e_spacing_y=1, e_spacing_x=0,
                 smoothing_num=1, alpha=None, i_func=None, config=None):
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
        @param row_remove: Alternatively, which rows to remove from dataset
        @type rows: Union[None, list, set]
        @param thin_data_points: Use every nth datapoint, useful for extremely dense datasets
        @type thin_data_points: int
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
        self.datnum = datnum
        self.datname = datname
        self.dc_num = dc_num
        self.dc_name = dc_name
        self.datdf = datdf
        self.dt = dt
        self.dat = self.set_dat(datnum, datname, datdf, verbose=False, config=config)
        self.dc = self.set_dc(dc_num, dc_name, datdf, verbose=False, config=config)
        self.i_func = i_func
        self.rows = rows
        self.row_remove = row_remove
        self.thin_data_points = thin_data_points
        self.i_spacing_y = i_spacing_y
        self.i_spacing_x = i_spacing_x
        self.e_spacing_y = e_spacing_y
        self.e_spacing_x = e_spacing_x
        self.smoothing_num = smoothing_num
        self.alpha = alpha

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
            C.print_verbose(f'WARNING[DatMeta]: More info required to set_dat: (datnum, datname, datdf) = \n'
                            f'[{datnum}, {datname}, {datdf}]', verbose)
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


class InDepthData(object):
    class Plot(object):
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

    @classmethod
    def get_default_plot_list(cls):
        show_plots = {
            'i_sense': False,  # waterfall w/fit
            'i_sense_raw': False,  # waterfall raw
            'i_sense_avg': True,  # averaged_i_sense
            'i_sense_avg_others': False,  # forced fits
            'entr': False,  # waterfall w/fit
            'entr_raw': False,  # waterfall raw
            'avg_entr': True,  # averaged_entr
            'avg_entr_others': False,  # forced fits
            'int_ent': True,  # integrated_entr
            'int_ent_others': False,  # forced fits
            'tables': False  # Fit info tables
        }
        return show_plots

    @staticmethod
    def get_dat_setup(datnum, set_name='Jan20_gamma'):
        """
        Neatening up where I store all the dat setup info

        @param datnum: datnum to load
        @type datnum: int
        """

        # Make variables accessible outside this function

        if set_name.lower() == 'jan20':
            # [1533, 1501]
            datdf = DF.DatDF(dfname='Apr20')
            if datnum == 1533:
                meta = DatMeta(1533, 'digamma_quad',
                               1529, 'base', datdf,
                               rows=list(range(0, 22, 5)),
                               thin_data_points=10,
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
                               thin_data_points=10,
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
            datdf = DF.DatDF(dfname='Apr20')
            if datnum == 1492:
                meta = DatMeta(1492, 'base',
                               1529, 'base', datdf,
                               rows=None,
                               row_remove={2, 6, 7, 9, 13, 14, 15, 21},
                               thin_data_points=50,
                               i_spacing_y=-2,
                               e_spacing_y=-3,
                               smoothing_num=1,
                               alpha=CU.get_alpha(0.82423, 100),
                               i_func=T.i_sense_digamma_quad
                               )
            elif datnum == 1495:
                meta = DatMeta(1495, 'base',
                               1529, 'base', datdf,
                               rows=None,
                               row_remove={0, 1, 3, 9, 13, 15},
                               thin_data_points=50,
                               i_spacing_y=-2,
                               e_spacing_y=-3,
                               smoothing_num=1,
                               alpha=CU.get_alpha(0.82423, 100),
                               i_func=T.i_sense_digamma_quad
                               )
            elif datnum == 1498:
                meta = DatMeta(1498, 'base',
                               1529, 'base', datdf,
                               rows=None,
                               row_remove={3, 4, 11},
                               thin_data_points=50,
                               i_spacing_y=-2,
                               e_spacing_y=-3,
                               smoothing_num=1,
                               alpha=CU.get_alpha(0.82423, 100),
                               i_func=T.i_sense_digamma_quad
                               )
            elif datnum == 1501:
                meta = DatMeta(1501, 'base',
                               1529, 'base', datdf,
                               rows=None,
                               row_remove={0, 3, 4},
                               thin_data_points=50,
                               i_spacing_y=-2,
                               e_spacing_y=-3,
                               smoothing_num=1,
                               alpha=CU.get_alpha(0.82423, 100),
                               i_func=T.i_sense_digamma_quad
                               )
            elif datnum == 1504:
                meta = DatMeta(1504, 'base',
                               1529, 'base', datdf,
                               rows=None,
                               row_remove={3, 6, 7},
                               thin_data_points=50,
                               i_spacing_y=-2,
                               e_spacing_y=-3,
                               smoothing_num=1,
                               alpha=CU.get_alpha(0.82423, 100),
                               i_func=T.i_sense_digamma_quad
                               )
            elif datnum == 1507:
                meta = DatMeta(1507, 'base',
                               1529, 'base', datdf,
                               rows=None,
                               row_remove={0, 4, 5, 6, 8},
                               thin_data_points=50,
                               i_spacing_y=-2,
                               e_spacing_y=-3,
                               smoothing_num=1,
                               alpha=CU.get_alpha(0.82423, 100),
                               i_func=T.i_sense_digamma_quad
                               )
            elif datnum == 1510:
                meta = DatMeta(1510, 'base',
                               1529, 'base', datdf,
                               rows=None,
                               row_remove={3, 4, 8, 10, 11},
                               thin_data_points=50,
                               i_spacing_y=-2,
                               e_spacing_y=-3,
                               smoothing_num=1,
                               alpha=CU.get_alpha(0.82423, 100),
                               i_func=T.i_sense_digamma_quad
                               )
            elif datnum == 1513:
                meta = DatMeta(1513, 'base',
                               1529, 'base', datdf,
                               rows=None,
                               row_remove={0, 2, 3, 4, 8, 10, 11},
                               thin_data_points=50,
                               i_spacing_y=-2,
                               e_spacing_y=-3,
                               smoothing_num=1,
                               alpha=CU.get_alpha(0.82423, 100),
                               i_func=T.i_sense_digamma_quad
                               )
            elif datnum == 1516:
                meta = DatMeta(1516, 'base',
                               1529, 'base', datdf,
                               rows=None,
                               row_remove={0, 3},
                               thin_data_points=50,
                               i_spacing_y=-2,
                               e_spacing_y=-3,
                               smoothing_num=1,
                               alpha=CU.get_alpha(0.82423, 100),
                               i_func=T.i_sense_digamma_quad
                               )
            elif datnum == 1519:
                meta = DatMeta(1519, 'base',
                               1529, 'base', datdf,
                               rows=None,
                               row_remove={3, 4, 8, 11},
                               thin_data_points=50,
                               i_spacing_y=-2,
                               e_spacing_y=-3,
                               smoothing_num=1,
                               alpha=CU.get_alpha(0.82423, 100),
                               i_func=T.i_sense_digamma_quad
                               )
            elif datnum == 1522:
                meta = DatMeta(1522, 'base',
                               1529, 'base', datdf,
                               rows=None,
                               row_remove={0, 1, 2, 8, 11},
                               thin_data_points=50,
                               i_spacing_y=-2,
                               e_spacing_y=-3,
                               smoothing_num=1,
                               alpha=CU.get_alpha(0.82423, 100),
                               i_func=T.i_sense_digamma_quad
                               )
            elif datnum == 1525:
                meta = DatMeta(1525, 'base',
                               1529, 'base', datdf,
                               rows=None,
                               row_remove={0, 1, 2, 4, 8, 9, 10, 11},
                               thin_data_points=50,
                               i_spacing_y=-2,
                               e_spacing_y=-3,
                               smoothing_num=1,
                               alpha=CU.get_alpha(0.82423, 100),
                               i_func=T.i_sense_digamma_quad
                               )
            elif datnum == 1528:
                meta = DatMeta(1528, 'base',
                               1529, 'base', datdf,
                               rows=None,
                               row_remove={0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11},
                               thin_data_points=50,
                               i_spacing_y=-2,
                               e_spacing_y=-3,
                               smoothing_num=1,
                               alpha=CU.get_alpha(0.82423, 100),
                               i_func=T.i_sense_digamma_quad
                               )
            else:
                raise ValueError(f'setup data for [{datnum}] does not exist in set [{set_name}]')

        elif set_name.lower() == 'jan20_gamma_2':
            datdf = DF.DatDF(dfname='Apr20')
            datnums = InDepthData.get_datnums(set_name)
            # Set defaults
            metas = [DatMeta(datnum=None, datname='digamma',
                             # Don't want to load all dats unless necessary because it's slow
                             dc_num=1529, dc_name='base', datdf=datdf,
                             rows=None,
                             row_remove=None,
                             thin_data_points=50,
                             i_spacing_y=-2,
                             e_spacing_y=-3,
                             smoothing_num=1,
                             alpha=CU.get_alpha(0.82423, 100),
                             i_func=T.i_sense_digamma_quad) for num in datnums]
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
            from src.Configs import Sep19Config
            sep19_config_switcher = CU.switch_config_decorator_maker(Sep19Config)
            datdf = CU.wrapped_call(sep19_config_switcher, (lambda: DF.DatDF(dfname='Apr20')))
            datnums = InDepthData.get_datnums(set_name)
            if datnum in datnums:
                # Set defaults
                metas = [DatMeta(datnum=None, datname='base',
                                 # Don't want to load all dats unless necessary because it's slow
                                 dc_num=1945, dc_name='base', datdf=datdf,
                                 rows=list(range(0, 21)),
                                 row_remove=None,
                                 thin_data_points=10,
                                 i_spacing_y=-2,
                                 i_spacing_x=-0.3,
                                 e_spacing_y=-3,
                                 e_spacing_x=0,
                                 smoothing_num=1,
                                 alpha=CU.get_alpha(23.85, 50),
                                 i_func=T.i_sense_digamma_quad,
                                 config=Sep19Config) for num in datnums]
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
            mar19_config_switcher = CU.switch_config_decorator_maker(Mar19Config)
            datdf = CU.wrapped_call(mar19_config_switcher, (lambda: DF.DatDF(dfname='Apr20')))
            datnums = InDepthData.get_datnums(set_name)
            if datnum in datnums:
                # Set defaults
                metas = [DatMeta(datnum=None, datname='base', datdf=datdf,
                                 # Don't want to load all dats unless necessary because it's slow
                                 dt=get_mar19_dt(750 / 50 * np.sqrt(2)),  # no single DCbias dat for Mar19
                                 rows=None,
                                 row_remove=None,
                                 thin_data_points=1,
                                 i_spacing_y=-2,
                                 i_spacing_x=-0.3,
                                 e_spacing_y=-3,
                                 e_spacing_x=0,
                                 smoothing_num=1,
                                 alpha=CU.get_alpha(6.727, 100),
                                 i_func=T.i_sense_digamma_quad,
                                 config=Mar19Config) for num in datnums]
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
            print(f'WARNING[InDepthData.get_datnums]: There are no datnums for set [{set_name}]')
            datnums = None
        return datnums

    def __init__(self, datnum, plots_to_show, set_name='Jan20_gamma', run_fits=True, show_plots=True):

        # region Data Setup
        self.datnum = datnum
        self.set_name = set_name
        self.setup_meta = InDepthData.get_dat_setup(datnum, set_name=set_name)
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
        self.x = CU.bin_data(dat.Data.x_array, self.setup_meta.thin_data_points)
        self.dx = np.abs(self.x[-1] - self.x[0]) / len(self.x)
        rows = self.setup_meta.rows
        if rows is None:
            print(f'For dat[{self.datnum}] - Loading all data rows')
            self.y_isense = CU.bin_data(dat.Data.i_sense, self.setup_meta.thin_data_points)
            self.y_entr = CU.bin_data(dat.Entropy.entr, self.setup_meta.thin_data_points)
        else:
            print(f'For dat[{self.datnum}] - Loading data rows: {rows}')
            self.y_isense = np.array([CU.bin_data(dat.Data.i_sense[i], self.setup_meta.thin_data_points) for i in rows])
            self.y_entr = np.array([CU.bin_data(dat.Entropy.entr[i], self.setup_meta.thin_data_points) for i in rows])

    def plot_all_plots(self):
        show_plots = self.plots_to_show
        self.plot_i_sense_by_row(raw=show_plots['i_sense_raw'], smoothed=show_plots['i_sense'],
                                 show_tables=show_plots['tables'])
        self.plot_entropy_by_row(raw=show_plots['entr_raw'], smoothed=show_plots['entr'],
                                 show_tables=show_plots['tables'])
        self.plot_average_i_sense(avg=show_plots['i_sense_avg'], others=show_plots['i_sense_avg_others'],
                                  show_tables=show_plots['tables'])
        self.plot_average_entropy(avg=show_plots['avg_entr'], others=show_plots['i_sense_avg_others'],
                                  show_tables=show_plots['tables'])
        self.plot_integrated(avg=show_plots['int_ent'], others=show_plots['int_ent_others'])

    def run_all_fits(self, i_params=None, i_func=None, e_params=None):
        self.i_params, self.i_func, self.i_fits = self.fit_isenses(params=i_params, func=i_func)
        self.e_params, self.e_fits = self.fit_entropys(params=e_params)

        self.make_averages()

        self.i_avg_fit = self._i_avg_fit(params=i_params, func=i_func)

        self.e_avg_fit = self._e_avg_fit(params=e_params)
        self.e_ln2_fit = self._e_ln2_fit()
        self.e_avg_dt_ln2 = self._e_avg_dt_ln2()  # Not really fitting, just setting dT for scaling differently

        self.i_amp_ln2_fit = self._i_amp_ln2_fit()  # Has to go after e_avg_fit

    def _i_avg_fit(self, params=None, func=None):
        if params is None:
            params = [self.i_params[0]]
        if func is None:
            func = self.i_func
        params = CU.ensure_params_list(params, self.i_avg, verbose=True)
        fit = FitData(T.transition_fits(self.x, self.i_avg, params=params, func=func)[0])
        self.i_avg_fit = fit
        return fit

    def _e_avg_fit(self, params=None):
        if params is None:
            params = [self.e_params[0]]
        dc = self.setup_meta.dc
        dat = self.setup_meta.dat
        params = CU.ensure_params_list(params, self.e_avg, verbose=True)
        fit = FitData(E.entropy_fits(self.x, self.e_avg, params=params)[0],
                      dt=self.setup_meta.dt,
                      amp=self.i_avg_fit.amp
                      )
        self.e_avg_fit = fit
        return fit

    def _e_ln2_fit(self):
        params = CU.edit_params(self.e_avg_fit.params, 'dS', np.log(2), vary=False)
        dc = self.setup_meta.dc
        dat = self.setup_meta.dat
        params = CU.ensure_params_list(params, self.e_avg, verbose=False)
        fit = FitData(E.entropy_fits(self.x, self.e_avg, params=params)[0],
                      dt=self.setup_meta.dt,
                      amp=self.i_avg_fit.amp
                      )
        self.e_ln2_fit = fit
        return fit

    def _e_avg_dt_ln2(self):
        """Doesn't actually need to do any fitting, just copying and then changing dT in the FitData object"""
        fake_fit = copy.deepcopy(self.e_avg_fit)
        fake_fit.dt = self.e_avg_fit.scaling_dt * self.e_avg_fit.integrated[-1] / np.log(2)
        fake_fit.set_sf()
        self.e_avg_dt_ln2 = fake_fit
        return fake_fit

    def _i_amp_ln2_fit(self):
        new_amp = self.i_avg_fit.amp * self.e_avg_fit.integrated[-1] / np.log(2)
        params = CU.edit_params(self.i_avg_fit.params, 'amp', new_amp, vary=False)
        for par in params:
            params[par].vary = False
        params = CU.ensure_params_list(params, self.i_avg, verbose=False)
        fit = FitData(T.transition_fits(self.x, self.i_avg, params=params, func=self.i_func)[0])
        self.i_amp_ln2_fit = fit
        return fit

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

    def plot_i_sense_by_row(self, raw=False, smoothed=True, show_tables=False, sub_poly=True):
        meta = self.setup_meta
        if raw is True:
            fig, axs = PF.make_axes(1, single_fig_size=self.fig_size)
            ax = axs[0]
            x, y = self.x, self.y_isense
            if sub_poly is True:
                x, y = sub_poly_from_data(x, y, self.i_fits)
            PF.waterfall_plot(x, y, ax=ax, y_spacing=meta.i_spacing_y, x_spacing=meta.i_spacing_x,
                              every_nth=1,
                              plot_args={'s': 1},
                              ptype='scatter', label=True, cmap_name=self.cmap_name, index=meta.rows)
            PF.ax_setup(ax, f'I_sense data for dat[{meta.dat.datnum}]', meta.dat.Logs.x_label, 'I_sense /nA',
                        legend=True)
            PF.add_standard_fig_info(fig)

        if smoothed is True:
            fig, axs = PF.make_axes(1, single_fig_size=self.fig_size)
            ax = axs[0]
            if meta.smoothing_num > 1:
                ysmooth = savgol_filter(self.y_isense, meta.smoothing_num, 1)
            else:
                ysmooth = self.y_isense
            xi = (CU.get_data_index(self.x, self.i_avg_fit.mid - self.view_width),
                  CU.get_data_index(self.x, self.i_avg_fit.mid + self.view_width))
            x, y = self.x[xi[0]:xi[1]], ysmooth[:, xi[0]:xi[1]]
            if sub_poly is True:
                x, y = sub_poly_from_data(x, y, self.i_fits)
            y_add, x_add = PF.waterfall_plot(x, y, ax=ax,
                                             y_spacing=meta.i_spacing_y,
                                             x_spacing=meta.i_spacing_x,
                                             every_nth=1, plot_args={'s': 1}, ptype='scatter', label=True,
                                             cmap_name=self.cmap_name, index=meta.rows)
            y_fits = np.array([fit.eval(x=self.x[xi[0]:xi[1]]) for fit in self.i_fits])
            x, y_fits = self.x[xi[0]:xi[1]], y_fits
            if sub_poly is True:
                x, y_fits = sub_poly_from_data(x, y_fits, self.i_fits)
            PF.waterfall_plot(x, y_fits, ax=ax, y_add=y_add, x_add=x_add, color='C3', ptype='plot')
            PF.ax_setup(ax, f'Smoothed I_sense data for dat[{meta.dat.datnum}]\nwith fits', meta.dat.Logs.x_label,
                        'I_sense /nA',
                        legend=True)
            PF.add_standard_fig_info(fig)

            if show_tables is True:
                df = CU.fit_info_to_df(self.i_fits, uncertainties=self.uncertainties, sf=3, index=meta.rows)
                PF.plot_df_table(df, title=f'I_sense_fit info for dat[{meta.dat.datnum}]')

    def plot_average_i_sense(self, avg=True, others=False, show_tables=False, sub_poly=True):
        meta = self.setup_meta
        if avg is True:
            fig, axs = PF.make_axes(1, single_fig_size=self.fig_size)

            ax = axs[0]
            # PF.display_1d(self.x, self.i_avg, ax, scatter=True, label='Averaged data')
            xi = (
                CU.get_data_index(self.x, self.i_avg_fit.mid - self.view_width),
                CU.get_data_index(self.x, self.i_avg_fit.mid + self.view_width))
            x, y = self.x[xi[0]:xi[1]], self.i_avg[xi[0]:xi[1]]
            if sub_poly is True:
                x, y = sub_poly_from_data(x, y, self.i_avg_fit.fit)
            PF.display_1d(x, y, ax, scatter=True, label='self.i_avg')
            # ax.plot(self.x_i_fit_avg, i_fit_avg.best_fit, c='C3', label='Best fit')

            x, y_fit = self.i_avg_fit.x, self.i_avg_fit.best_fit
            if sub_poly is True:
                x, y_fit = sub_poly_from_data(x, y_fit, self.i_avg_fit.fit)
            ax.plot(x, y_fit, c='C3', label='i_avg_fit.best_fit')
            PF.ax_setup(ax, f'Dat[{meta.dat.datnum}]:Averaged data with fit', meta.dat.Logs.x_label, 'I_sense /nA',
                        legend=True)

            if show_tables is True:
                df = CU.fit_info_to_df([self.i_avg_fit.fit], uncertainties=self.uncertainties, sf=3, index=meta.rows)
                df.pop('index')
                PF.plot_df_table(df, title=f'Dat[{meta.dat.datnum}]:I_sense fit values no additional forcing')

            PF.add_standard_fig_info(fig)

        if others is True:
            fig, axs = PF.make_axes(2, single_fig_size=self.fig_size)
            ax = axs[0]
            # PF.display_1d(self.x, self.i_avg, ax, scatter=True, label='Averaged data')
            PF.display_1d(self.x, self.i_avg, ax, scatter=True, label='self.i_avg')
            # ax.plot(x_i_fit_avg, i_fit_ln2.best_fit, c='C3', label='Ln(2) amplitude fit')
            ax.plot(self.i_amp_ln2_fit.x, self.i_amp_ln2_fit.best_fit, c='C3', label='i_amp_ln2_fit.best_fit')
            PF.ax_setup(ax, f'Dat[{meta.dat.datnum}]:Averaged I_sense data with\nwith amp forced s.t. int_dS = Ln(2)',
                        meta.dat.Logs.x_label, 'I_sense /nA', legend=True)
            PF.add_standard_fig_info(fig)

            if show_tables is True:
                df = CU.fit_info_to_df([self.i_amp_ln2_fit.fit], uncertainties=self.uncertainties, sf=3,
                                       index=meta.rows)
                df.pop('index')
                PF.plot_df_table(df,
                                 title=f'Dat[{meta.dat.datnum}]:I_sense fit values with amp forced s.t. int_dS = Ln(2)')

    def plot_entropy_by_row(self, raw=False, smoothed=True, show_tables=False):
        meta = self.setup_meta

        if raw is True:
            fig, axs = PF.make_axes(1, single_fig_size=self.fig_size)
            ax = axs[0]
            y_add, x_add = PF.waterfall_plot(self.x, self.y_entr, ax=ax, y_spacing=meta.e_spacing_y,
                                             x_spacing=meta.e_spacing_x,
                                             every_nth=1, plot_args={'s': 1}, ptype='scatter', label=True,
                                             cmap_name=self.cmap_name, index=meta.rows)
            PF.ax_setup(ax, f'Entropy_r data for dat[{meta.dat.datnum}]', meta.dat.Logs.x_label, 'Entr /nA',
                        legend=True)
            PF.add_standard_fig_info(fig)

        if smoothed is True:
            fig, axs = PF.make_axes(1, single_fig_size=self.fig_size)
            ax = axs[0]
            if meta.smoothing_num > 1:
                ysmooth = savgol_filter(self.y_entr, meta.smoothing_num, 1)
            else:
                ysmooth = self.y_entr
            xi = (CU.get_data_index(self.x, self.i_avg_fit.mid - self.view_width),
                  CU.get_data_index(self.x, self.i_avg_fit.mid + self.view_width))
            y_add, x_add = PF.waterfall_plot(self.x[xi[0]:xi[1]], ysmooth[:, xi[0]:xi[1]], ax=ax,
                                             y_spacing=meta.e_spacing_y,
                                             x_spacing=meta.e_spacing_x,
                                             every_nth=1,
                                             plot_args={'s': 1}, ptype='scatter', label=True, cmap_name=self.cmap_name,
                                             index=meta.rows)
            y_fits = np.array([fit.eval(x=self.x[xi[0]:xi[1]]) for fit in self.e_fits])
            PF.waterfall_plot(self.x[xi[0]:xi[1]], y_fits, ax=ax, y_add=y_add, x_add=x_add, color='C3', ptype='plot')
            PF.ax_setup(ax, f'Dat[{meta.dat.datnum}]:Smoothed entropy_r data\nwith fits', meta.dat.Logs.x_label,
                        'Entr /nA', legend=True)
            PF.add_standard_fig_info(fig)

            if show_tables is True:
                df = CU.fit_info_to_df(self.e_fits, uncertainties=self.uncertainties, sf=3, index=meta.rows)
                PF.plot_df_table(df, title=f'Entropy_R_fit info for dat[{meta.dat.datnum}]')
            PF.add_standard_fig_info(fig)

    def plot_average_entropy(self, avg=True, others=False, show_tables=False):
        meta = self.setup_meta
        if avg is True:
            fig, axs = PF.make_axes(1, single_fig_size=self.fig_size)
            ax = axs[0]
            # PF.display_1d(self.x, e_y_avg, ax, scatter=True, label='Averaged data')
            PF.display_1d(self.x, self.e_avg, ax, scatter=True, label='e_avg')
            # ax.plot(x_e_fit_avg, e_fit_avg.best_fit, c='C3', label='Best fit')
            ax.plot(self.e_avg_fit.x, self.e_avg_fit.best_fit, c='C3', label='e_avg_fit.best_fit')
            PF.ax_text(ax, f'dT={self.e_avg_fit.scaling_dt * meta.alpha / Const.kb * 1000:.3f}mK', loc=(0.02, 0.6))
            PF.ax_setup(ax, f'Dat[{meta.dat.datnum}]:Averaged Entropy R data with fit', meta.dat.Logs.x_label,
                        'Entropy R /nA',
                        legend=True)

            if show_tables is True:
                df = CU.fit_info_to_df([self.e_avg_fit.fit], uncertainties=self.uncertainties, sf=3, index=meta.rows)
                df.pop('index')
                PF.plot_df_table(df, title=f'Dat[{meta.dat.datnum}]:Entropy R fit values with no additional forcing')

            PF.add_standard_fig_info(fig)

        if others is True:
            fig, axs = PF.make_axes(2, single_fig_size=self.fig_size)
            # region Forced to dS = Ln2
            ax = axs[0]
            # PF.display_1d(self.x, e_y_avg, ax, scatter=True, label='Averaged data')
            PF.display_1d(self.x, self.e_avg, ax, scatter=True, label='e_avg')
            # ax.plot(x_e_fit_avg, e_fit_ln2.best_fit, c='C3', label='Ln(2) fit')
            ax.plot(self.e_ln2_fit.x, self.e_ln2_fit.best_fit, c='C3', label='e_ln2_fit.best_fit')
            PF.ax_text(ax, f'dT={self.e_ln2_fit.scaling_dt * meta.alpha / Const.kb * 1000:.3f}mK', loc=(0.02, 0.6))
            PF.ax_setup(ax, f'Dat[{meta.dat.datnum}]:Averaged Entropy R data with Ln(2) fit', meta.dat.Logs.x_label,
                        'Entropy R /nA',
                        legend=True)
            PF.add_standard_fig_info(fig)

            if show_tables is True:
                df = CU.fit_info_to_df([self.e_ln2_fit.fit], uncertainties=self.uncertainties, sf=3, index=meta.rows)
                df.pop('index')
                PF.plot_df_table(df, title=f'Dat[{meta.dat.datnum}]:Entropy R fit values with dS forced to Ln2')

            # region Forced dT s.t. int_data dS = ln2
            ax = axs[1]
            # PF.display_1d(self.x, e_y_avg, ax, scatter=True, label='Averaged data')
            PF.display_1d(self.x, self.e_avg, ax, scatter=True, label='e_avg')
            # ax.plot(x_e_fit_avg, e_fit_dt_ln2.best_fit, c='C3', label='dT forced fit')
            ax.plot(self.e_avg_dt_ln2.x, self.e_avg_dt_ln2.best_fit, c='C3', label='e_avg_dt_ln2.best_fit')
            PF.ax_text(ax, f'dT={self.e_avg_dt_ln2.scaling_dt * meta.alpha / Const.kb * 1000:.3f}mK', loc=(0.02, 0.6))
            PF.ax_setup(ax, f'Dat[{meta.dat.datnum}]:Averaged Entropy R data\nwith dT forced s.t. int_data dS = Ln2',
                        meta.dat.Logs.x_label, 'Entropy R /nA',
                        legend=True)
            PF.add_standard_fig_info(fig)

            if show_tables is True:
                df = CU.fit_info_to_df([self.e_avg_dt_ln2.fit], uncertainties=self.uncertainties, sf=3, index=meta.rows)
                df.pop('index')
                PF.plot_df_table(df,
                                 title=f'Dat[{meta.dat.datnum}]:Entropy R fit values\nwith dT forced s.t. int_data dS = Ln2')
                # endregion
                PF.add_standard_fig_info(fig)

    def plot_integrated(self, avg=True, others=False):
        meta = self.setup_meta
        if avg is True:
            # region dT from DCbias, amp from I_sense, also int of e_fit_avg
            fig, axs = PF.make_axes(1, single_fig_size=self.fig_size)
            ax = axs[0]
            # PF.display_1d(self.x, int_avg, ax, label='Averaged data')
            PF.display_1d(self.e_avg_fit.x, self.e_avg_fit.integrated, ax, label='e_avg_int')
            # ax.plot(x_e_fit_avg, int_of_fit, c='C3', label='integrated best fit')
            ax.plot(self.e_avg_fit.x, self.e_avg_fit.fit_integrated, c='C3', label='int_of_fit')
            PF.ax_setup(ax, f'Dat[{meta.dat.datnum}]:Integrated Entropy\ndT from DCbias for data and fit',
                        meta.dat.Logs.x_label,
                        'Entropy /kB')
            _add_ln3_ln2(ax)
            _add_peak_final_text(ax, self.e_avg_fit.integrated, self.e_avg_fit.fit_integrated)
            ax.legend(loc='lower right')
            PF.ax_text(ax, f'dT = {self.e_avg_fit.scaling_dt * meta.alpha / Const.kb * 1000:.3f}mK\n'
                           f'amp = {self.i_avg_fit.amp:.3f}nA\n'
                           f'int_avg dS={self.e_avg_fit.integrated[-1] / np.log(2):.3f}kBLn2\n'
                           f'int_of_fit dS={self.e_avg_fit.fit_integrated[-1] / np.log(2):.3f}kBLn2',
                       loc=(0.02, 0.7), fontsize=8)

            PF.add_standard_fig_info(fig)
            # endregion
        if others is True:
            fig, axs = PF.make_axes(3, single_fig_size=self.fig_size)
            # region dT adjusted s.t. integrated_data has dS = ln2, fit with that dt forced then integrated
            ax = axs[0]
            # PF.display_1d(self.x, int_avg_dt_ln2, ax, label='Averaged data')
            PF.display_1d(self.e_avg_dt_ln2.x, self.e_avg_dt_ln2.integrated, ax, label='e_avg_dt_ln2')
            # ax.plot(x_e_fit_avg, int_of_fit_dt_ln2, c='C3', label='integrated fit\nwith dT forced')
            ax.plot(self.e_avg_dt_ln2.x, self.e_avg_dt_ln2.fit_integrated, c='C3', label='fit')
            PF.ax_setup(ax, f'Dat[{meta.dat.datnum}]:Integrated Entropy\ndT forced s.t. int_ds=Ln2',
                        meta.dat.Logs.x_label,
                        'Entropy /kB')
            _add_ln3_ln2(ax)
            _add_peak_final_text(ax, self.e_avg_dt_ln2.integrated, self.e_avg_dt_ln2.fit_integrated)
            ax.legend(loc='lower right')
            PF.ax_text(ax, f'dT of forced fit={self.e_avg_dt_ln2.scaling_dt * meta.alpha / Const.kb * 1000:.3f}mK\n'
                           f'amp = {self.i_avg_fit.amp:.3f}nA\n'
                           f'int_avg_dt_ln2 dS={self.e_avg_dt_ln2.integrated[-1] / np.log(2):.3f}kBLn2\n'
                           f'int_fit_dt_ln2 dS={self.e_avg_dt_ln2.fit_integrated[-1] / np.log(2):.3f}kBLn2',
                       loc=(0.02, 0.7), fontsize=8)


def get_exp_df(exp_name='mar19', dfname='Apr20'):
    if exp_name.lower() == 'mar19':
        from src.Configs import Mar19Config
        mar19_config_switcher = CU.switch_config_decorator_maker(Mar19Config)
        datdf = CU.wrapped_call(mar19_config_switcher, (lambda: DF.DatDF(dfname=dfname)))
        return datdf
    elif exp_name.lower() == 'sep19':
        from src.Configs import Sep19Config
        sep19_config_switcher = CU.switch_config_decorator_maker(Sep19Config)
        datdf = CU.wrapped_call(sep19_config_switcher, (lambda: DF.DatDF(dfname=dfname)))
        return datdf
    elif exp_name.lower() == 'jan20':
        from src.Configs import Jan20Config
        jan20_config_switcher = CU.switch_config_decorator_maker(Jan20Config)
        datdf = CU.wrapped_call(jan20_config_switcher, (lambda: DF.DatDF(dfname=dfname)))
        return datdf
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


def sub_poly_from_data(x, y, fits):
    """
    Subtracts polynomial terms from data if they exist (i.e. will sub up to quadratic term)
    @param x: x data
    @type x: np.ndarray
    @param y: y data
    @type y: np.ndarray
    @param fits: lm fit(s) which has best values for up to quad term (const, lin, quad)
    @type fits: Union[list[lm.model.ModelResult], lm.model.ModelResult]
    @return: tuple of x, y or list of x, y tuples
    @rtype: Union[tuple[np.ndarray], list[tuple[np.ndarray]]]
    """

    def _sub_1d(x1d, y1d, fit1d):
        mid = fit1d.best_values.get('mid', 0)
        const = fit1d.best_values.get('const', 0)
        lin = fit1d.best_values.get('lin', 0)
        quad = fit1d.best_values.get('quad', 0)

        x1d = x1d - mid
        subber = lambda x, y: y - quad * x ** 2 - lin * x - const
        y1d = subber(x1d, y1d)
        return x1d, y1d

    x = np.asarray(x)
    y = np.asarray(y)
    assert x.ndim == 1
    assert y.ndim in [1, 2]
    assert isinstance(fits, (lm.model.ModelResult, list, tuple, np.ndarray))
    if y.ndim == 2:
        x = np.array([x] * y.shape[0])
        if not isinstance(fits, (list, tuple, np.ndarray)):
            fits = [fits] * y.shape[0]
        return x[0], np.array([_sub_1d(x1, y1, fit1)[1] for x1, y1, fit1 in zip(x, y, fits)])

    elif y.ndim == 1:
        return _sub_1d(x, y, fits)


def _plot_rows_of_data(idd, row_range=(0, 5)):
    """For looking at bump in charge sensor data, looking at average of different chunks
    of rows to see if the bump is always there"""
    ax = plt.gca()
    ax.cla()
    cut = row_range
    ys = idd.y_isense[cut[0]:cut[1]]
    x = idd.x - np.average(idd.x)
    y = CU.average_data(ys, [fit.best_values['mid'] for fit in idd.i_fits[cut[0]:cut[1]]])[0].astype(np.float32)
    y = y - idd.i_avg_fit.quad * x ** 2 - idd.i_avg_fit.lin * x - idd.i_avg_fit.const
    pars = idd.i_avg_fit.params
    pars = CU.edit_params(pars, ['const', 'lin', 'quad', 'mid'], [0, 0, 0, 0], [True, True, True, True])
    fit = idd.i_avg_fit.fit.model.fit(y, params=pars, x=x, nan_policy='omit')
    ax.scatter(x, y, s=1)
    ax.plot(fit.userkws['x'], fit.best_fit, color='C3', label='Fit')
    PF.add_scatter_label(f'{cut[0]} to {cut[1]}')
    ax.legend()
    ax.legend().set_title('Data rows')
    PF.ax_setup(ax, f'Dat{idd.datnum}: I_sense poly subtracted\nLooking for bump on right side', 'Gate /mV',
                'Current (offset) /nA')
    plt.tight_layout()


def _plot_entropy_vs_gamma(IDDs, fig_title='Jan20 Entropy vs Gamma', gate_fn=(lambda x: getattr(x, 'fdacs')[4])):
    """
    4 axes: a few i_avg_fit.data/fit, a few e_avg_fit.integrated, entropy vs gamma, entropy vs gate where gate obtained
    applying gate_fn to dat.Logs

    @param IDDs:
    @type IDDs: list[InDepthData]
    @param fig_title:
    @type fig_title: str
    @param gate_fn: function to apply to dat.Logs to get the value of gate responsible for coupling
    (lambda x: getattr(x, 'dacs')[13]) for sep19
    @type gate_fn: func
    @return: None
    @rtype: None
    """

    fig, axs = PF.make_axes(4)
    fig.suptitle(fig_title)
    ax = axs[0]
    for idd in IDDs:
        idd.Plot.plot_avg_i(idd, ax, True, True, True, True)
    PF.ax_setup(ax, 'Avg i_sense', 'Plunger /mV', 'Current /nA', legend=True)
    ax.legend().set_title('Dat')
    ax = axs[1]
    for idd in IDDs:
        idd.Plot.plot_int_e(idd, ax)
    PF.ax_setup(ax, 'Integrated_entropy', 'Plunger /mV', 'Entropy /kB', legend=True)
    ax.legend().set_title('Dat')
    axs[0].legend().set_title('Dat')
    ax = axs[2]
    xs = [idd.i_avg_fit.g for idd in IDDs]
    ys = [idd.e_avg_fit.integrated[-1] for idd in IDDs]
    ax.scatter(xs, ys, s=3)
    for x, y, idd in zip(xs, ys, IDDs):
        ax.text(x, y, f'{idd.datnum}', fontsize=6)

    PF.ax_setup(ax, 'Integrated entropy vs Gamma', 'Gamma /mV', 'Entropy /kB')
    ax = axs[3]
    xs = [gate_fn(idd.setup_meta.dat.Logs) for idd in IDDs]
    ys = [idd.e_avg_fit.integrated[-1] for idd in IDDs]
    ax.scatter(xs, ys, s=3)
    for x, y, idd in zip(xs, ys, IDDs):
        ax.text(x, y, f'{idd.datnum}', fontsize=6)

    PF.ax_setup(ax, 'Integrated entropy vs Coupling gate', 'Coupling Gate /mV', 'Entropy /kB')
    PF.add_standard_fig_info(fig)


def _i_sense_data_to_yigal(IDDs, show=True, save_to_file=False):
    if show is True:
        fig, axs = PF.make_axes(num=len(IDDs), single_fig_size=IDDs[0].fig_size,
                                plt_kwargs={'sharex': False, 'sharey': True})
    else:
        axs = np.zeros(len(IDDs))

    for idd, ax in zip(IDDs, axs):
        x = idd.x - idd.i_avg_fit.mid
        y = idd.i_avg
        x, y = map(np.array, zip(*([[x1, y1] for x1, y1 in zip(x, y) if not np.isnan(y1)])))
        subber = lambda x, y: y - idd.i_avg_fit.quad * x ** 2 - idd.i_avg_fit.lin * x - idd.i_avg_fit.const
        y = subber(x, y)
        x_fit = idd.i_avg_fit.x - idd.i_avg_fit.mid
        y_fit = subber(x_fit, idd.i_avg_fit.best_fit)

        if show is True:
            PF.display_1d(x, y, ax=ax, scatter=True)
            ax.plot(x_fit, y_fit, label='fit', color='C3')
            PF.ax_setup(ax, f'Dat[{idd.datnum}]: I_avg minus polynomial terms', 'Gate /mV', 'Current (offset to 0) /nA',
                        legend=True)
            PF.ax_text(ax, f'amp={idd.i_avg_fit.amp:.3f}nA\n'
                           f'theta={idd.i_avg_fit.theta:.3f}mV\n'
                           f'gamma={idd.i_avg_fit.g:.3f}mV', loc=(0.02, 0.05))
            # plt.tight_layout(rect=[0, 0.05, 1, 1])

        if save_to_file is True:
            datapath = os.path.normpath(
                r'D:\OneDrive\UBC LAB\My work\My_Papers\2Resources\Equations_and_Graphs\Yigal\i_sense_data_to_send')
            data = np.array([x, y])
            filepath = os.path.join(datapath, f'dat[{idd.datnum}]')
            sio.savemat(filepath + '.mat', {'x': x, 'i_sense': y})
            np.savetxt(filepath + '.csv', data, delimiter=',')



def get_fit_params(IDDs):
    idd # type: InDepthData
    i_df = CU.fit_info_to_df([idd.i_avg_fit.fit for idd in IDDs], uncertainties=True, index=[f'{idd.setup_meta.datdf.config_name[0:5]}[{idd.datnum}]' for idd in IDDs])
    e_df = CU.fit_info_to_df([idd.e_avg_fit.fit for idd in IDDs], uncertainties=True, index=[f'{idd.setup_meta.datdf.config_name[0:5]}[{idd.datnum}]' for idd in IDDs])
    PF.plot_df_table(i_df, 'I_sense Fit Info for Sep19 and Jan20')
    PF.plot_df_table(e_df, 'Entropy Fit Info for Sep19 and Jan20')
    print(i_df)
    print(e_df)

sep_datdf = get_exp_df('sep19')

mar_datdf = get_exp_df('mar19')

jan_datdf = get_exp_df('jan20')


if __name__ == '__main__':
    plots = {
        'i_sense': True,  # waterfall w/fit
        'i_sense_raw': False,  # waterfall raw
        'i_sense_avg': True,  # averaged_i_sense
        'i_sense_avg_others': False,  # forced fits
        'entr': False,  # waterfall w/fit
        'entr_raw': False,  # waterfall raw
        'avg_entr': False,  # averaged_entr
        'avg_entr_others': False,  # forced fits
        'int_ent': False,  # integrated_entr
        'int_ent_others': False,  # forced fits
        'tables': True  # Fit info tables
    }
    run_mar = False
    run_sep = True
    run_jan1 = False
    run_jan2 = False


    if run_mar is True:
        m_datnums = InDepthData.get_datnums('mar19_gamma_entropy')
        m_IDDs = [InDepthData(num, plots_to_show=plots, set_name='mar19_gamma_entropy', run_fits=False, show_plots=False)
                  for
                  num in m_datnums]
        m_idd_dict = dict(zip(m_datnums, m_IDDs))
        e_params = m_IDDs[0].setup_meta.dat.Entropy.avg_params
        e_params = CU.edit_params(e_params, 'const', 0, False)
        for idd in m_IDDs:
            idd.run_all_fits(e_params=e_params)


    if run_sep is True:
        # f = InDepthData(1563, plots_to_show=plots, set_name='Jan20_gamma_2', run_fits=True, config=None)

        # dc = C.DatHandler.get_dat(1947, 'base', sep_datdf, config=Sep19Config)
        # dat = C.DatHandler.get_dat(2713, 'base', sep_datdf, config=Sep19Config)
        # dc.DCbias.plot_self(dc, dat)
        s_datnums = InDepthData.get_datnums('sep19_gamma')
        s_IDDs = [InDepthData(num, plots_to_show=plots, set_name='Sep19_gamma', run_fits=False, show_plots=False) for
                  num in
                  s_datnums[0:8]]
        s_idd_dict = dict(zip(s_datnums[0:8], s_IDDs))
        e_params = s_IDDs[0].setup_meta.dat.Entropy.avg_params
        cols = ['datnum', 'offset']
        data = [[]]
        for f in s_IDDs:
            # f.plot_integrated(avg=True, others=False)
            # i_params = f.i_fits[0].params
            # i_params = CU.edit_params(i_params, 'g', 0, False)
            f.fit_isenses()
            i_params = [fit.params for fit in f.i_fits]
            i_params = [CU.edit_params(param, 'theta', 27.3, False) for param in i_params]
            f.fit_isenses(params=i_params)
            f.make_averages()
            # offset = np.nanmean([f.e_avg[CU.get_data_index(f.x, -2000):CU.get_data_index(f.x, -1500)], f.e_avg[CU.get_data_index(f.x, 1500):CU.get_data_index(f.x, 2100)]] )
            # f.y_entr = f.y_entr - offset
            e_params = CU.edit_params(e_params, 'const', 0, False)
            # data.append([f.datnum, offset])
            #
            # f.run_all_fits(i_params=i_params, e_params=e_params)
            f.run_all_fits(i_params=None, e_params=e_params)
            # f.plot_integrated(avg=True, others=False)
            # f.plot_all_plots()

        # df = pd.DataFrame(data, columns=cols)
        # print(df)
        # PF.plot_df_table(df, sig_fig=4)

    if run_jan1 is True:
        j1_datnums = InDepthData.get_datnums('jan20_gamma')
        j1_IDDs = [InDepthData(num, plots, set_name='jan20_gamma', run_fits=False, show_plots=False) for num in
                   j1_datnums]
        j1_idd_dict = dict(zip(j1_datnums, j1_IDDs))
        e_params = j1_IDDs[0].setup_meta.dat.Entropy.avg_params
        for idd in j1_IDDs:
            idd.fit_isenses()
            i_params = idd.i_fits[0].params
            if idd.datnum not in [1492, 1495]:
                i_params = CU.edit_params(i_params, 'theta', 0.9765, False)
            else:
                i_params = CU.edit_params(i_params, 'g', 0, False)
            e_params = CU.edit_params(e_params, 'const', 0, False)
            e_params = CU.edit_params(e_params, 'mid', idd.i_fits[0].best_values['mid'])
            idd.run_all_fits(i_params=i_params, e_params=e_params)
        # Per IDD fixes.

    if run_jan2 is True:
        j2_datnums = InDepthData.get_datnums('jan20_gamma_2')
        j2_IDDs = [InDepthData(num, plots, set_name='jan20_gamma_2', run_fits=False, show_plots=False) for num in
                   j2_datnums]
        j2_idd_dict = dict(zip(j2_datnums, j2_IDDs))
        e_params = j2_IDDs[0].setup_meta.dat.Entropy.avg_params
        for idd in j2_IDDs:
            idd.fit_isenses()
            i_params = idd.i_fits[0].params
            if idd.datnum not in [1533, 1536]:
                i_params = CU.edit_params(i_params, 'theta', 0.976, False)
            else:
                i_params = CU.edit_params(i_params, 'g', 0, False)
            e_params = CU.edit_params(e_params, 'const', 0, False)
            e_params = CU.edit_params(e_params, 'mid', idd.i_fits[0].best_values['mid'])
            idd.run_all_fits(i_params=i_params, e_params=e_params)


