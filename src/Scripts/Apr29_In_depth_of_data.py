from src.Scripts.StandardImports import *
import src.DatCode.Entropy as E
import src.DatCode.Transition as T
from scipy.signal import savgol_filter
import copy
import lmfit as lm

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


# def _load_dat_if_necessary(global_name: str, datnum: int, datdf:DF.DatDF, datname: str = 'base'):
#     if global_name in globals():
#         old_dat = globals()[global_name]
#         try:
#             if old_dat.datnum == datnum and\
#                     old_dat.datname == datname and\
#                     old_dat.dfname == datdf.name and\
#                     old_dat.config_name == datdf.config_name:
#                 print(f'Reusing {global_name}[{datnum}]')
#                 return old_dat
#             else:
#                 print(f'[{global_name}] does not match requested dat')
#         except Exception as e:
#             print(f'Error when comparing existing [{global_name}] to given datnum, datdf, datname')
#             pass
#     print(f'[{global_name}] = dat[{datnum}] loaded with make_dat_standard')
#     new_dat = make_dat_standard(datnum, datname, dfoption='load', datdf=datdf)
#     globals()['global_name'] = new_dat
#     return new_dat


class DatMeta(object):
    """
    Object to store info about how to use data from given dataset
    """

    def __init__(self, datnum=None, datname=None, dc_num=None, dc_name=None, datdf=None, rows=None, row_remove=None, thin_data_points=1,
                 i_spacing_y=-1,
                 e_spacing_y=1, smoothing_num=1, alpha=None, i_func=None):
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
        @param rows: which rows of data to use from dataset
        @type rows: Union[None, list, set]
        @param row_remove: Alternatively, which rows to remove from dataset
        @type rows: Union[None, list, set]
        @param thin_data_points: Use every nth datapoint, useful for extremely dense datasets
        @type thin_data_points: int
        @param i_spacing_y: relative spacing of charge sensor data in waterfall plot
        @type i_spacing_y: float
        @param e_spacing_y: relative spacing of entropy data in waterfall plot
        @type e_spacing_y: float
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
        self.dat = self.set_dat(datnum, datname, datdf, verbose=False)
        self.dc = self.set_dc(dc_num, dc_name, datdf, verbose=False)
        self.i_func = i_func
        self.rows = rows
        self.row_remove = row_remove
        self.thin_data_points = thin_data_points
        self.i_spacing_y = i_spacing_y
        self.e_spacing_y = e_spacing_y
        self.smoothing_num = smoothing_num
        self.alpha = alpha

        if self.row_remove is not None:  # make self.rows only contain desired rows
            self.set_rows()


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

    def set_dat(self, datnum=None, datname=None, datdf=None, verbose=True):
        datnum = datnum if datnum is not None else self.datnum
        datname = datname if datname is not None else self.datname
        datdf = datdf if datdf is not None else self.datdf
        if None not in [datnum, datname, datdf]:
            self.datnum = datnum
            self.datname = datname
            self.datdf = datdf
            self.dat = C.DatHandler.get_dat(datnum, datname, datdf)
        else:
            C.print_verbose(f'WARNING[DatMeta]: More info required to set_dat: (datnum, datname, datdf) = \n'
                            f'[{datnum}, {datname}, {datdf}]', verbose)
            return None
        return self.dat

    def set_dc(self, dc_num=None, dc_name=None, datdf=None, verbose=True):
        datnum = dc_num if dc_num is not None else self.dc_num
        datname = dc_name if dc_name is not None else self.dc_name
        datdf = datdf if datdf is not None else self.datdf
        if None not in [datnum, datname, datdf]:
            self.dc_num = dc_num
            self.dc_name = dc_name
            self.datdf = datdf
            self.dc = C.DatHandler.get_dat(dc_num, dc_name, datdf)
        else:
            C.print_verbose(f'WARNING[DatMeta]: More info required to set_dc: (dc_num, dc_name, datdf) = \n'
                            f'[{datnum}, {datname}, {datdf}]', verbose)
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
        return np.abs(x[-1]-x[0])/len(x)

    @property
    def sf(self):
        sf = None
        if None not in [self.scaling_dt, self.scaling_amp, self.dx]:
            sf = E.scaling(self.scaling_dt, self.scaling_amp, self.dx)

        if self._sf is not None and self._sf == sf:
            return sf
        elif self._sf is not None and self._sf != sf:
            print(f'WARNING[FitData]: sf from dt, amp, dx = [{CU.sig_fig(sf,3)}], stored sf is [{CU.sig_fig(self._sf)}]'
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
        return np.nancumsum(self.data)*self.sf

    @property
    def fit_integrated(self):
        return np.nancumsum(self.best_fit)*self.sf

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
        return np.nancumsum(self.data)*sf


class InDepthData(object):

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
            datnums = [1492, 1495, 1498, 1501, 1504, 1507, 1510, 1513, 1516, 1519, 1522, 1525, 1528]
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
                meta = DatMeta(0, 'base',
                               1513, 'base', datdf,
                               rows=None,
                               row_remove={3, 4, 8, 10, 11},
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
            datnums = [1533, 1536, 1539, 1542, 1545, 1548, 1551, 1554, 1557, 1560, 1563, 1566]
            # Set defaults
            metas = [DatMeta(datnum=None, datname='digamma',  # Don't want to load all dats unless necessary because it's slow
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
                metas[datnum].row_remove = {1,3}
            elif datnum == 1536:
                metas[datnum].set_dat(datnum)
                metas[datnum].row_remove = {}
            elif datnum == 1539:
                metas[datnum].set_dat(datnum)
                metas[datnum].row_remove = {}
            elif datnum == 1542:
                metas[datnum].set_dat(datnum)
                metas[datnum].row_remove = {}
            elif datnum == 1545:
                metas[datnum].set_dat(datnum)
                metas[datnum].row_remove = {}
            elif datnum == 1548:
                metas[datnum].set_dat(datnum)
                metas[datnum].row_remove = {}
            elif datnum == 1551:
                metas[datnum].set_dat(datnum)
                metas[datnum].row_remove = {}
            elif datnum == 1554:
                metas[datnum].set_dat(datnum)
                metas[datnum].row_remove = {}
            elif datnum == 1557:
                metas[datnum].set_dat(datnum)
                metas[datnum].row_remove = {}
            elif datnum == 1560:
                metas[datnum].set_dat(datnum)
                metas[datnum].row_remove = {}
            elif datnum == 1563:
                metas[datnum].set_dat(datnum)
                metas[datnum].row_remove = {}
            elif datnum == 1566:
                metas[datnum].set_dat(datnum)
                metas[datnum].row_remove = {}
            else:
                raise ValueError(f'setup data for [{datnum}] does not exist in set [{set_name}]')
            meta = metas[datnum]


        else:
            raise ValueError(f'Set [{set_name}] does not exist in get_dat_setup')

        return meta

    def __init__(self, datnum, plots_to_show, set_name='Jan20_gamma', run_fits=True, show_plots=True):

        # region Data Setup
        self.datnum = datnum
        self.set_name = set_name
        self.setup_meta = InDepthData.get_dat_setup(datnum, set_name=set_name)
        # endregion

        # region Data
        dat = self.setup_meta.dat
        self.x = dat.Data.x_array[::self.setup_meta.thin_data_points]
        self.dx = np.abs(self.x[-1] - self.x[0]) / len(self.x)
        rows = self.setup_meta.rows
        if rows is None:
            print(f'For dat[{self.datnum}] - Loading all data rows')
            self.y_isense = dat.Data.i_sense[:, ::self.setup_meta.thin_data_points]
            self.y_entr = dat.Entropy.entr[:, ::self.setup_meta.thin_data_points]
        else:
            print(f'For dat[{self.datnum}] - Loading data rows: {rows}')
            self.y_isense = np.array([dat.Data.i_sense[i, ::self.setup_meta.thin_data_points] for i in rows])
            self.y_entr = np.array([dat.Entropy.entr[i, ::self.setup_meta.thin_data_points] for i in rows])
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
            self.run_all_fits()

        # region Plot related
        self.plots_to_show = plots_to_show
        self.view_width = 1000
        self.fig_size = (5, 5)
        self.cmap_name = 'tab10'
        # endregion

        # region Table related
        self.uncertainties = True  # Whether to show uncertainties in tables or not
        # endregion

        if show_plots is True and run_fits is True:
            self.plot_all_plots()

    def plot_all_plots(self):
        show_plots = self.plots_to_show
        self.plot_i_sense_by_row(raw=show_plots['i_sense_raw'], smoothed=show_plots['i_sense'], show_tables=show_plots['tables'])
        self.plot_entropy_by_row(raw=show_plots['entr_raw'], smoothed=show_plots['entr'], show_tables=show_plots['tables'])
        self.plot_average_i_sense(avg=show_plots['i_sense_avg'], others=show_plots['i_sense_avg_others'], show_tables=show_plots['tables'])
        self.plot_average_entropy(avg=show_plots['avg_entr'], others=show_plots['i_sense_avg_others'], show_tables=show_plots['tables'])
        self.plot_integrated(avg=show_plots['int_ent'], others=show_plots['int_ent_others'])

    def run_all_fits(self):
        self.i_params, self.i_func, self.i_fits = self.fit_isenses()
        self.e_params, self.e_fits = self.fit_entropys()

        self.make_averages()

        self.i_avg_fit = FitData(T.transition_fits(self.x, self.i_avg, params=[self.i_params[0]], func=self.i_func)[0])

        self.e_avg_fit = self._e_avg_fit()
        self.e_ln2_fit = self._e_ln2_fit()
        self.e_avg_dt_ln2 = self._e_avg_dt_ln2()  # Not really fitting, just setting dT for scaling differently

        self.i_amp_ln2_fit = self._i_amp_ln2_fit()  # Has to go after e_avg_fit

    def _e_avg_fit(self):
        dc = self.setup_meta.dc
        dat = self.setup_meta.dat
        fit = FitData(E.entropy_fits(self.x, self.e_avg, params=[self.e_params[0]])[0],
                      dt=dc.DCbias.get_dt_at_current(dat.Instruments.srs1.out / 50 * np.sqrt(2)),
                      amp=self.i_avg_fit.amp
                      )
        return fit

    def _e_ln2_fit(self):
        params = CU.edit_params(self.e_avg_fit.params, 'dS', np.log(2), vary=False)
        dc = self.setup_meta.dc
        dat = self.setup_meta.dat
        fit = FitData(E.entropy_fits(self.x, self.e_avg, params=[params])[0],
                      dt=dc.DCbias.get_dt_at_current(dat.Instruments.srs1.out / 50 * np.sqrt(2)),
                      amp=self.i_avg_fit.amp
                      )
        return fit

    def _e_avg_dt_ln2(self):
        """Doesn't actually need to do any fitting, just copying and then changing dT in the FitData object"""
        fake_fit = copy.deepcopy(self.e_avg_fit)
        fake_fit.dt = self.e_avg_fit.scaling_dt*self.e_avg_fit.integrated[-1]/np.log(2)
        fake_fit.set_sf()
        return fake_fit

    def _i_amp_ln2_fit(self):
        new_amp = self.i_avg_fit.amp * self.e_avg_fit.integrated[-1]/np.log(2)
        params = CU.edit_params(self.i_avg_fit.params, 'amp', new_amp, vary=False)
        for par in params:
            params[par].vary = False
        fit = FitData(T.transition_fits(self.x, self.i_avg, params=[params], func=self.i_func)[0])
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
        return params, func, T.transition_fits(x, data, params=params, func=func)

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

        return params, E.entropy_fits(x, data, params=params)

    def plot_i_sense_by_row(self, raw=True, smoothed=True, show_tables=True):
        meta = self.setup_meta
        if raw is True:
            fig, axs = PF.make_axes(1, single_fig_size=self.fig_size)
            ax = axs[0]
            PF.waterfall_plot(self.x, self.y_isense, ax=ax, y_spacing=meta.i_spacing_y, x_add=0, every_nth=1,
                                             plot_args={'s': 1},
                                             ptype='scatter', label=True, cmap_name=self.cmap_name, index=meta.rows)
            PF.ax_setup(ax, f'I_sense data for dat[{meta.dat.datnum}]', meta.dat.Logs.x_label, 'I_sense /nA', legend=True)
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
            y_add, x_add = PF.waterfall_plot(self.x[xi[0]:xi[1]], ysmooth[:, xi[0]:xi[1]], ax=ax, y_spacing=meta.i_spacing_y,
                                             x_add=0,
                                             every_nth=1, plot_args={'s': 1}, ptype='scatter', label=True,
                                             cmap_name=self.cmap_name, index=meta.rows)
            y_fits = np.array([fit.eval(x=self.x[xi[0]:xi[1]]) for fit in self.i_fits])
            PF.waterfall_plot(self.x[xi[0]:xi[1]], y_fits, ax=ax, y_add=y_add, x_add=x_add, color='C3', ptype='plot')
            PF.ax_setup(ax, f'Smoothed I_sense data for dat[{meta.dat.datnum}]\nwith fits', meta.dat.Logs.x_label, 'I_sense /nA',
                        legend=True)
            PF.add_standard_fig_info(fig)

            if show_tables is True:
                df = CU.fit_info_to_df(self.i_fits, uncertainties=self.uncertainties, sf=3, index=meta.rows)
                PF.plot_df_table(df, title=f'I_sense_fit info for dat[{meta.dat.datnum}]')
    
    def plot_average_i_sense(self, avg=True, others=True, show_tables=True):
        meta = self.setup_meta
        if avg is True:
            fig, axs = PF.make_axes(1, single_fig_size=self.fig_size)

            ax = axs[0]
            # PF.display_1d(self.x, self.i_avg, ax, scatter=True, label='Averaged data')
            xi = (
                CU.get_data_index(self.x, self.i_avg_fit.mid - self.view_width),
                CU.get_data_index(self.x, self.i_avg_fit.mid + self.view_width))
            PF.display_1d(self.x[xi[0]:xi[1]], self.i_avg[xi[0]:xi[1]], ax, scatter=True, label='self.i_avg')
            # ax.plot(self.x_i_fit_avg, i_fit_avg.best_fit, c='C3', label='Best fit')
            ax.plot(self.i_avg_fit.x, self.i_avg_fit.best_fit, c='C3', label='i_avg_fit.best_fit')
            PF.ax_setup(ax, f'Dat[{meta.dat.datnum}]:Averaged data with fit', meta.dat.Logs.x_label, 'I_sense /nA', legend=True)

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
                df = CU.fit_info_to_df([self.i_amp_ln2_fit.fit], uncertainties=self.uncertainties, sf=3, index=meta.rows)
                df.pop('index')
                PF.plot_df_table(df, title=f'Dat[{meta.dat.datnum}]:I_sense fit values with amp forced s.t. int_dS = Ln(2)')

    def plot_entropy_by_row(self, raw=True, smoothed=True, show_tables=True):
        meta = self.setup_meta
        
        if raw is True:
            fig, axs = PF.make_axes(1, single_fig_size=self.fig_size)
            ax = axs[0]
            y_add, x_add = PF.waterfall_plot(self.x, self.y_entr, ax=ax, y_spacing=meta.e_spacing_y, x_add=0,
                                             every_nth=1, plot_args={'s': 1}, ptype='scatter', label=True,
                                             cmap_name=self.cmap_name, index=meta.rows)
            PF.ax_setup(ax, f'Entropy_r data for dat[{meta.dat.datnum}]', meta.dat.Logs.x_label, 'Entr /nA', legend=True)
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
            y_add, x_add = PF.waterfall_plot(self.x[xi[0]:xi[1]], ysmooth[:, xi[0]:xi[1]], ax=ax, y_spacing=meta.e_spacing_y,
                                             x_add=0,
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
    
    def plot_average_entropy(self, avg=True, others=True, show_tables=True):
        meta = self.setup_meta
        if avg is True:
            fig, axs = PF.make_axes(1, single_fig_size=self.fig_size)
            ax = axs[0]
            # PF.display_1d(self.x, e_y_avg, ax, scatter=True, label='Averaged data')
            PF.display_1d(self.x, self.e_avg, ax, scatter=True, label='e_avg')
            # ax.plot(x_e_fit_avg, e_fit_avg.best_fit, c='C3', label='Best fit')
            ax.plot(self.e_avg_fit.x, self.e_avg_fit.best_fit, c='C3', label='e_avg_fit.best_fit')
            PF.ax_text(ax, f'dT={self.e_avg_fit.scaling_dt * meta.alpha * 1000:.3f}mK', loc=(0.02, 0.6))
            PF.ax_setup(ax, f'Dat[{meta.dat.datnum}]:Averaged Entropy R data with fit', meta.dat.Logs.x_label, 'Entropy R /nA',
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
            PF.ax_text(ax, f'dT={self.e_ln2_fit.scaling_dt * meta.alpha * 1000:.3f}mK', loc=(0.02, 0.6))
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
            PF.ax_text(ax, f'dT={self.e_avg_dt_ln2.scaling_dt * meta.alpha * 1000:.3f}mK', loc=(0.02, 0.6))
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
        
    def plot_integrated(self, avg=True, others=True):
        meta = self.setup_meta
        if avg is True:
            # region dT from DCbias, amp from I_sense, also int of e_fit_avg
            fig, axs = PF.make_axes(1, single_fig_size=self.fig_size)
            ax = axs[0]
            # PF.display_1d(self.x, int_avg, ax, label='Averaged data')
            PF.display_1d(self.e_avg_fit.x, self.e_avg_fit.integrated, ax, label='e_avg_int')
            # ax.plot(x_e_fit_avg, int_of_fit, c='C3', label='integrated best fit')
            ax.plot(self.e_avg_fit.x, self.e_avg_fit.fit_integrated, c='C3', label='int_of_fit')
            PF.ax_setup(ax, f'Dat[{meta.dat.datnum}]:Integrated Entropy\ndT from DCbias for data and fit', meta.dat.Logs.x_label,
                        'Entropy /kB')
            _add_ln3_ln2(ax)
            _add_peak_final_text(ax, self.e_avg_fit.integrated, self.e_avg_fit.fit_integrated)
            ax.legend(loc='lower right')
            PF.ax_text(ax, f'dT = {self.e_avg_fit.scaling_dt * meta.alpha * 1000:.3f}mK\n'
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
            PF.ax_setup(ax, f'Dat[{meta.dat.datnum}]:Integrated Entropy\ndT forced s.t. int_ds=Ln2', meta.dat.Logs.x_label,
                        'Entropy /kB')
            _add_ln3_ln2(ax)
            _add_peak_final_text(ax, self.e_avg_dt_ln2.integrated, self.e_avg_dt_ln2.fit_integrated)
            ax.legend(loc='lower right')
            PF.ax_text(ax, f'dT of forced fit={self.e_avg_dt_ln2.scaling_dt * meta.alpha * 1000:.3f}mK\n'
                           f'amp = {self.i_avg_fit.amp:.3f}nA\n'
                           f'int_avg_dt_ln2 dS={self.e_avg_dt_ln2.integrated[-1] / np.log(2):.3f}kBLn2\n'
                           f'int_fit_dt_ln2 dS={self.e_avg_dt_ln2.fit_integrated[-1] / np.log(2):.3f}kBLn2',
                       loc=(0.02, 0.7), fontsize=8)


if __name__ == '__main__':
    # plots = InDepthData.get_default_plot_list()  # Can get from here too
    plots = {
            'i_sense': True,  # waterfall w/fit
            'i_sense_raw': False,  # waterfall raw
            'i_sense_avg': True,  # averaged_i_sense
            'i_sense_avg_others': False,  # forced fits
            'entr': False,  # waterfall w/fit
            'entr_raw': False,  # waterfall raw
            'avg_entr': True,  # averaged_entr
            'avg_entr_others': False,  # forced fits
            'int_ent': False,  # integrated_entr
            'int_ent_others': False,  # forced fits
            'tables': False  # Fit info tables
        }


    # a = InDepthData(1522, plots_to_show=plots, set_name='Jan20_gamma', run_fits=True, show_plots=False)
    # print('here')
    b = InDepthData(1533, plots_to_show=plots, set_name='Jan20_gamma_2', run_fits=True)

run = False
if run is True:
    pass
    # datdf = DF.DatDF(dfname='Apr20')
    # assert datdf.config_name == 'Jan20Config'
    #
    # # region Setup data to look at
    # # region Fake load just so variables exist before setting in get_dat_setup()
    # rows = []
    # every_nth = 1
    # from_to = (None, None)
    # thin_data_points = 1
    # i_spacing_y = 1
    # e_spacing_y = 1
    # smoothing_num = 1
    # view_width = 1
    # beta = 1  # Theta in mV at min point of DCbias / 100mK in K.  T(K) = beta*theta(mV)
    # # endregion
    #
    # # [1492, 1495, 1498, 1501, 1504, 1507, 1510, 1513, 1516, 1519, 1522, 1525, 1528]
    # get_dat_setup(1522)
    # dat = _load_dat_if_necessary('dat', dat.datnum, datdf, dat.datname)  # Just to stop script complaining
    # dc = _load_dat_if_necessary('dc', dc.datnum, datdf, dc.datname)  # Just to stop script complaining
    # # endregion
    #
    # view_width = 20  # overrides view_width
    # fig_size = (5, 5)  # Size of each axes in a figure
    # show_plots = {
    #     'i_sense': False,  # waterfall w/fit
    #     'i_sense_raw': False,  # waterfall raw
    #     'i_sense_avg': True,  # averaged_i_sense
    #     'i_sense_avg_others': False,  # forced fits
    #     'entr': False,  # waterfall w/fit
    #     'entr_raw': False,  # waterfall raw
    #     'avg_entr': False,  # averaged_entr
    #     'avg_entr_others': False,  # forced fits
    #     'int_ent': False,  # integrated_entr
    #     'int_ent_others': False,  # forced fits
    #     'tables': False  # Fit info tables
    # }
    # cmap_name = 'tab10'
    # uncertainties = True  # Whether to show uncertainties in tables or not
    # use_existing_fits = False  # Use existing fits for waterfall plots and as starting params?
    # transition_fit_func = T.i_sense_digamma_quad  # What func to fit to i_sense data
    #
    # # region Select data to look at
    # assert float(dat.version) >= 1.3  # D.Dat.version  # Make sure loaded dat is up to date
    # assert dat.Entropy.version == E.Entropy.version  # Make sure loaded dat has most up to date Entropy
    # x = dat.Data.x_array[::thin_data_points]
    # dx = (x[-1] - x[0]) / len(x)
    # if rows is None:
    #     print(f'Loading every {every_nth} data row from row '
    #           f'{from_to[0] if from_to[0] is not None else 0} to '
    #           f'{from_to[1] if from_to[1] is not None else "end"}')
    #     y_isense = dat.Data.i_sense[from_to[0]:from_to[1]:every_nth, ::thin_data_points]
    #     y_entr = dat.Entropy.entr[from_to[0]:from_to[1]:every_nth, ::thin_data_points]
    # else:
    #     print(f'Loading data rows: {rows}')
    #     y_isense = np.array([dat.Data.i_sense[i, ::thin_data_points] for i in rows])
    #     y_entr = np.array([dat.Entropy.entr[i, ::thin_data_points] for i in rows])
    # # endregion
    #
    # # region Get or make fit data
    # if use_existing_fits is True:  # Match up to rows of data chosen
    #     # region Get Existing fits
    #     fits_isense = dat.Transition._full_fits[from_to[0]:from_to[1]:every_nth]
    #     fits_entr = dat.Entropy._full_fits[from_to[0]:from_to[1]:every_nth]
    #     # endregion
    # else:  # Make new fits
    #     # region Make Transition fits
    #     params = T.get_param_estimates(x, y_isense)
    #     for par in params:
    #         T._append_param_estimate_1d(par, ['g', 'quad'])
    #
    #     # Edit fit pars here
    #     params = [CU.edit_params(par, param_name='g', value=0, vary=True, min_val=-10, max_val=None) for par in params]
    #
    #     fits_isense = T.transition_fits(x, y_isense, params=params, func=transition_fit_func)
    #     # endregion
    #     # region Make Entropy fits
    #     mids = [fit.best_values['mid'] for fit in fits_isense]
    #     thetas = [fit.best_values['theta'] for fit in fits_isense]
    #     params = E.get_param_estimates(x, y_entr, mids, thetas)
    #
    #     # Edit fit pars here
    #     params = [CU.edit_params(par, param_name='const', value=0, vary=False, min_val=None, max_val=None) for par in
    #               params]
    #
    #     fits_entr = E.entropy_fits(x, y_entr, params=params)
    #     # endregion
    # # endregion
    #
    # # region Average of data being looked at ONLY
    # i_y_avg, _ = np.array(
    #     CU.average_data(y_isense, [CU.get_data_index(x, fit.best_values['mid']) for fit in fits_isense]))
    # e_y_avg, _ = np.array(
    #     CU.average_data(y_entr, [CU.get_data_index(x, fit.best_values['mid']) for fit in fits_isense]))
    # # endregion
    #
    # # region Fits to average of data being looked at ONLY
    # i_fit_avg = T.transition_fits(x, i_y_avg, params=[fits_isense[0].params], func=transition_fit_func)[0]
    # x_i_fit_avg = i_fit_avg.userkws['x']
    # e_fit_avg = E.entropy_fits(x, e_y_avg, params=[fits_entr[0].params])[0]
    # x_e_fit_avg = e_fit_avg.userkws['x']
    # # endregion
    #
    # # region Integrated Entropy with dT from DCbias amp from i_sense (standard)
    # dt = dc.DCbias.get_dt_at_current(dat.Instruments.srs1.out / 50 * np.sqrt(2))
    # sf = E.scaling(dt, i_fit_avg.best_values['amp'], dx)
    # int_avg = E.integrate_entropy_1d(e_y_avg, sf)
    # int_of_fit = E.integrate_entropy_1d(e_fit_avg.best_fit, sf)
    # # endregion
    #
    # # region E fit with dS forced = Ln2
    # params = CU.edit_params(e_fit_avg.params, 'dS', np.log(2), vary=False)
    # e_fit_ln2 = E.entropy_fits(x, e_y_avg, params=[params])[0]
    # # endregion
    #
    # # region E fit with dT forced s.t. int_data dS = Ln2
    # dt_ln2 = dt * int_avg[-1] / np.log(2)  # scaling prop to 1/dT
    # params = CU.edit_params(e_fit_avg.params, 'dT', dt / beta, False)
    # e_fit_dt_ln2 = E.entropy_fits(x, e_y_avg, params=[params])[0]
    # # endregion
    #
    # # region Integrated E of fit with dT forced s.t. int_data dS = Ln2
    # sf_dt_forced = E.scaling(dt_ln2, i_fit_avg.best_values['amp'], dx)
    # int_of_fit_dt_ln2 = E.integrate_entropy_1d(e_fit_dt_ln2.best_fit, sf_dt_forced)
    # int_avg_dt_ln2 = E.integrate_entropy_1d(e_y_avg, sf_dt_forced)
    # # endregion
    #
    # # region Integrated E of data and best fit with dT from E_avg fit
    # dt_from_fit = e_fit_avg.best_values['dT'] * beta
    # sf_dt_from_fit = E.scaling(dt_from_fit, i_fit_avg.best_values['amp'], dx)
    # int_avg_dt_from_fit = E.integrate_entropy_1d(e_y_avg, sf_dt_from_fit)
    # int_of_fit_dt_from_fit = E.integrate_entropy_1d(e_fit_avg.best_fit, sf_dt_from_fit)
    # # endregion
    #
    # # region I_sense with amp forced s.t. int_data dS = Ln2 with dT from DCbias
    # amp_forced_ln2 = i_fit_avg.best_values['amp'] * int_avg[-1] / np.log(2)
    # params = CU.edit_params(i_fit_avg.params, 'amp', amp_forced_ln2, vary=False)
    # i_fit_ln2 = T.transition_fits(x, i_y_avg, params=[params], func=transition_fit_func)[0]
    # # endregion
    #
    # # region I_sense with amp forced s.t. int_avg_fit dS = E_avg_fit dS with dT from E_avg fit.
    # amp_forced_fit_ds = i_fit_avg.best_values['amp'] * int_of_fit_dt_from_fit[-1] / e_fit_avg.best_values[
    #     'dS']  # sf prop to 1/amp
    # params = CU.edit_params(i_fit_avg.params, 'amp', amp_forced_fit_ds, vary=False)
    # i_fit_ds = T.transition_fits(x, i_y_avg, params=[params], func=transition_fit_func)[0]
    # # endregion
    #
    # # region Integrated E of data and best fit with dT from E_avg fit and amp s.t. int_avg_fit dS = E_avg_fit dS
    # # dt_from_fit = e_fit_avg.best_values['dT'] * beta  # Calculated above
    # sf_from_fit = E.scaling(dt_from_fit, amp_forced_fit_ds, dx)
    # int_avg_sf_from_fit = E.integrate_entropy_1d(e_y_avg, sf_from_fit)
    # int_of_fit_sf_from_fit = E.integrate_entropy_1d(e_fit_avg.best_fit, sf_from_fit)
    # # endregion
    #
    # #  PLOTTING BELOW HERE
    #
    # # region I_sense by row plots
    # if show_plots['i_sense_raw'] is True:
    #     fig, axs = PF.make_axes(1, single_fig_size=fig_size)
    #     ax = axs[0]
    #     y_add, x_add = PF.waterfall_plot(x, y_isense, ax=ax, y_spacing=i_spacing_y, x_add=0, every_nth=1,
    #                                      plot_args={'s': 1},
    #                                      ptype='scatter', label=True, cmap_name=cmap_name, index=rows)
    #     PF.ax_setup(ax, f'I_sense data for dat[{dat.datnum}]', dat.Logs.x_label, 'I_sense /nA', legend=True)
    #     PF.add_standard_fig_info(fig)
    #
    # if show_plots['i_sense'] is True:
    #     fig, axs = PF.make_axes(1, single_fig_size=fig_size)
    #     ax = axs[0]
    #     if smoothing_num > 1:
    #         ysmooth = savgol_filter(y_isense, smoothing_num, 1)
    #     else:
    #         ysmooth = y_isense
    #     xi = (CU.get_data_index(x, i_fit_avg.best_values['mid'] - view_width),
    #           CU.get_data_index(x, i_fit_avg.best_values['mid'] + view_width))
    #     y_add, x_add = PF.waterfall_plot(x[xi[0]:xi[1]], ysmooth[:, xi[0]:xi[1]], ax=ax, y_spacing=i_spacing_y, x_add=0,
    #                                      every_nth=1, plot_args={'s': 1}, ptype='scatter', label=True,
    #                                      cmap_name=cmap_name, index=rows)
    #     y_fits = np.array([fit.eval(x=x[xi[0]:xi[1]]) for fit in fits_isense])
    #     PF.waterfall_plot(x[xi[0]:xi[1]], y_fits, ax=ax, y_add=y_add, x_add=x_add, color='C3', ptype='plot')
    #     PF.ax_setup(ax, f'Smoothed I_sense data for dat[{dat.datnum}]\nwith fits', dat.Logs.x_label, 'I_sense /nA',
    #                 legend=True)
    #     PF.add_standard_fig_info(fig)
    #
    #     if show_plots['tables'] is True:
    #         df = CU.fit_info_to_df(fits_isense, uncertainties=uncertainties, sf=3, index=rows)
    #         PF.plot_df_table(df, title=f'I_sense_fit info for dat[{dat.datnum}]')
    # # endregion
    #
    # # region Average I_sense plots
    # if show_plots['i_sense_avg'] is True:
    #     # region No params forced
    #     fig, axs = PF.make_axes(1, single_fig_size=fig_size)
    #
    #     ax = axs[0]
    #     # PF.display_1d(x, i_y_avg, ax, scatter=True, label='Averaged data')
    #     xi = (
    #         CU.get_data_index(x, dat.Transition.mid - view_width),
    #         CU.get_data_index(x, dat.Transition.mid + view_width))
    #     PF.display_1d(x, i_y_avg, ax, scatter=True, label='i_y_avg')
    #     # ax.plot(x_i_fit_avg, i_fit_avg.best_fit, c='C3', label='Best fit')
    #     ax.plot(x_i_fit_avg, i_fit_avg.best_fit, c='C3', label='i_fit_avg.best_fit')
    #     PF.ax_setup(ax, f'Dat[{dat.datnum}]:Averaged data with fit', dat.Logs.x_label, 'I_sense /nA', legend=True)
    #
    #     if show_plots['tables'] is True:
    #         df = CU.fit_info_to_df([i_fit_avg], uncertainties=uncertainties, sf=3, index=rows)
    #         df.pop('index')
    #         PF.plot_df_table(df, title=f'Dat[{dat.datnum}]:I_sense fit values no additional forcing')
    #
    #     PF.add_standard_fig_info(fig)
    #     # endregion
    #
    # if show_plots['i_sense_avg_others'] is True:
    #     # region Amplitude forced s.t. integrated = Ln2 with dT from DCbias
    #     fig, axs = PF.make_axes(2, single_fig_size=fig_size)
    #     ax = axs[0]
    #     # PF.display_1d(x, i_y_avg, ax, scatter=True, label='Averaged data')
    #     PF.display_1d(x, i_y_avg, ax, scatter=True, label='i_y_avg')
    #     # ax.plot(x_i_fit_avg, i_fit_ln2.best_fit, c='C3', label='Ln(2) amplitude fit')
    #     ax.plot(x_i_fit_avg, i_fit_ln2.best_fit, c='C3', label='i_fit_ln2.best_fit')
    #     PF.ax_setup(ax, f'Dat[{dat.datnum}]:Averaged I_sense data with\nwith amp forced s.t. int_dS = Ln(2)',
    #                 dat.Logs.x_label, 'I_sense /nA', legend=True)
    #     PF.add_standard_fig_info(fig)
    #
    #     if show_plots['tables'] is True:
    #         df = CU.fit_info_to_df([i_fit_ln2], uncertainties=uncertainties, sf=3, index=rows)
    #         df.pop('index')
    #         PF.plot_df_table(df, title=f'Dat[{dat.datnum}]:I_sense fit values with amp forced s.t. int_dS = Ln(2)')
    #     # endregion
    #
    #     # region Amplitude forced s.t. integrated fit dS = fit dS (with dT from fit)
    #     ax = axs[1]
    #     # PF.display_1d(x, i_y_avg, ax, scatter=True, label='Averaged data')
    #     PF.display_1d(x, i_y_avg, ax, scatter=True, label='i_y_avg')
    #     # ax.plot(x_i_fit_avg, i_fit_ds.best_fit, c='C3', label='amp s.t.\nint_fit dS = fit_dS')
    #     ax.plot(x_i_fit_avg, i_fit_ds.best_fit, c='C3', label='i_fit_ds.best_fit')
    #     PF.ax_setup(ax,
    #                 f'Dat[{dat.datnum}]:Averaged I_sense data\nwith amp forced s.t. int_fit dS=fit dS\n(with dT from fit)',
    #                 dat.Logs.x_label,
    #                 'I_sense /nA', legend=True)
    #     PF.add_standard_fig_info(fig)
    #
    #     if show_plots['tables'] is True:
    #         df = CU.fit_info_to_df([i_fit_ds], uncertainties=uncertainties, sf=3, index=rows)
    #         df.pop('index')
    #         PF.plot_df_table(df,
    #                          title=f'Dat[{dat.datnum}]:I_sense fit values with amp forced s.t. int_fit dS=fit dS (with dT from fit)')
    #     # endregion
    #     PF.add_standard_fig_info(fig)
    # # endregion
    #
    # # region Entropy by row plots
    # if show_plots['entr_raw'] is True:
    #     fig, axs = PF.make_axes(1, single_fig_size=fig_size)
    #     ax = axs[0]
    #     y_add, x_add = PF.waterfall_plot(x, y_entr, ax=ax, y_spacing=e_spacing_y, x_add=0, every_nth=1,
    #                                      plot_args={'s': 1},
    #                                      ptype='scatter', label=True, cmap_name=cmap_name, index=rows)
    #     PF.ax_setup(ax, f'Entropy_r data for dat[{dat.datnum}]', dat.Logs.x_label, 'Entr /nA', legend=True)
    #     PF.add_standard_fig_info(fig)
    #
    # if show_plots['entr'] is True:
    #     fig, axs = PF.make_axes(1, single_fig_size=fig_size)
    #     ax = axs[0]
    #     if smoothing_num > 1:
    #         ysmooth = savgol_filter(y_entr, smoothing_num, 1)
    #     else:
    #         ysmooth = y_entr
    #     xi = (CU.get_data_index(x, i_fit_avg.best_values['mid'] - view_width),
    #           CU.get_data_index(x, i_fit_avg.best_values['mid'] + view_width))
    #     y_add, x_add = PF.waterfall_plot(x[xi[0]:xi[1]], ysmooth[:, xi[0]:xi[1]], ax=ax, y_spacing=e_spacing_y, x_add=0,
    #                                      every_nth=1,
    #                                      plot_args={'s': 1}, ptype='scatter', label=True, cmap_name=cmap_name,
    #                                      index=rows)
    #     y_fits = np.array([fit.eval(x=x[xi[0]:xi[1]]) for fit in fits_entr])
    #     PF.waterfall_plot(x[xi[0]:xi[1]], y_fits, ax=ax, y_add=y_add, x_add=x_add, color='C3', ptype='plot')
    #     PF.ax_setup(ax, f'Dat[{dat.datnum}]:Smoothed entropy_r data\nwith fits', dat.Logs.x_label,
    #                 'Entr /nA', legend=True)
    #     PF.add_standard_fig_info(fig)
    #
    #     if show_plots['tables'] is True:
    #         df = CU.fit_info_to_df(fits_entr, uncertainties=uncertainties, sf=3, index=rows)
    #         PF.plot_df_table(df, title=f'Entropy_R_fit info for dat[{dat.datnum}]')
    #     PF.add_standard_fig_info(fig)
    # # endregion
    #
    # # region Average Entropy Plots
    # if show_plots['avg_entr'] is True:
    #     # region No params forced
    #     fig, axs = PF.make_axes(1, single_fig_size=fig_size)
    #     ax = axs[0]
    #     # PF.display_1d(x, e_y_avg, ax, scatter=True, label='Averaged data')
    #     PF.display_1d(x, e_y_avg, ax, scatter=True, label='e_y_avg')
    #     # ax.plot(x_e_fit_avg, e_fit_avg.best_fit, c='C3', label='Best fit')
    #     ax.plot(x_e_fit_avg, e_fit_avg.best_fit, c='C3', label='e_fit_avg.best_fit')
    #     PF.ax_text(ax, f'dT={dt / beta * 1000:.3f}mK', loc=(0.02, 0.6))
    #     PF.ax_setup(ax, f'Dat[{dat.datnum}]:Averaged Entropy R data with fit', dat.Logs.x_label, 'Entropy R /nA',
    #                 legend=True)
    #
    #     if show_plots['tables'] is True:
    #         df = CU.fit_info_to_df([e_fit_avg], uncertainties=uncertainties, sf=3, index=rows)
    #         df.pop('index')
    #         PF.plot_df_table(df, title=f'Dat[{dat.datnum}]:Entropy R fit values with no additional forcing')
    #
    #     PF.add_standard_fig_info(fig)
    #     # endregion
    #
    # if show_plots['avg_entr_others'] is True:
    #     fig, axs = PF.make_axes(2, single_fig_size=fig_size)
    #     # region Forced to dS = Ln2
    #     ax = axs[0]
    #     # PF.display_1d(x, e_y_avg, ax, scatter=True, label='Averaged data')
    #     PF.display_1d(x, e_y_avg, ax, scatter=True, label='e_y_avg')
    #     # ax.plot(x_e_fit_avg, e_fit_ln2.best_fit, c='C3', label='Ln(2) fit')
    #     ax.plot(x_e_fit_avg, e_fit_ln2.best_fit, c='C3', label='e_fit_ln2.best_fit')
    #     PF.ax_text(ax, f'dT={dt / beta * 1000:.3f}mK', loc=(0.02, 0.6))
    #     PF.ax_setup(ax, f'Dat[{dat.datnum}]:Averaged Entropy R data with Ln(2) fit', dat.Logs.x_label, 'Entropy R /nA',
    #                 legend=True)
    #     PF.add_standard_fig_info(fig)
    #
    #     if show_plots['tables'] is True:
    #         df = CU.fit_info_to_df([e_fit_ln2], uncertainties=uncertainties, sf=3, index=rows)
    #         df.pop('index')
    #         PF.plot_df_table(df, title=f'Dat[{dat.datnum}]:Entropy R fit values with dS forced to Ln2')
    #     # endregion
    #
    #     # region Forced dT s.t. int_data dS = ln2
    #     ax = axs[1]
    #     # PF.display_1d(x, e_y_avg, ax, scatter=True, label='Averaged data')
    #     PF.display_1d(x, e_y_avg, ax, scatter=True, label='e_y_avg')
    #     # ax.plot(x_e_fit_avg, e_fit_dt_ln2.best_fit, c='C3', label='dT forced fit')
    #     ax.plot(x_e_fit_avg, e_fit_dt_ln2.best_fit, c='C3', label='e_fit_dt_ln2.best_fit')
    #     PF.ax_text(ax, f'dT={dt_ln2 / beta * 1000:.3f}mK', loc=(0.02, 0.6))
    #     PF.ax_setup(ax, f'Dat[{dat.datnum}]:Averaged Entropy R data\nwith dT forced s.t. int_data dS = Ln2',
    #                 dat.Logs.x_label, 'Entropy R /nA',
    #                 legend=True)
    #     PF.add_standard_fig_info(fig)
    #
    #     if show_plots['tables'] is True:
    #         df = CU.fit_info_to_df([e_fit_dt_ln2], uncertainties=uncertainties, sf=3, index=rows)
    #         df.pop('index')
    #         PF.plot_df_table(df, title=f'Dat[{dat.datnum}]:Entropy R fit values\nwith dT forced s.t. int_data dS = Ln2')
    #     # endregion
    #     PF.add_standard_fig_info(fig)
    # # endregion
    #
    # # region Integrated Entropy Plots
    # if show_plots['int_ent'] is True:
    #     # region dT from DCbias, amp from I_sense, also int of e_fit_avg
    #     fig, axs = PF.make_axes(1, single_fig_size=fig_size)
    #     ax = axs[0]
    #     # PF.display_1d(x, int_avg, ax, label='Averaged data')
    #     PF.display_1d(x, int_avg, ax, label='int_avg')
    #     # ax.plot(x_e_fit_avg, int_of_fit, c='C3', label='integrated best fit')
    #     ax.plot(x_e_fit_avg, int_of_fit, c='C3', label='int_of_fit')
    #     PF.ax_setup(ax, f'Dat[{dat.datnum}]:Integrated Entropy\ndT from DCbias for data and fit', dat.Logs.x_label,
    #                 'Entropy /kB')
    #     _add_ln3_ln2(ax)
    #     _add_peak_final_text(ax, int_avg, int_of_fit)
    #     ax.legend(loc='lower right')
    #     PF.ax_text(ax, f'dT = {dt / beta * 1000:.3f}mK\n'
    #                    f'amp = {i_fit_avg.best_values["amp"]:.3f}nA\n'
    #                    f'int_avg dS={int_avg[-1] / np.log(2):.3f}kBLn2\n'
    #                    f'int_of_fit dS={int_of_fit[-1] / np.log(2):.3f}kBLn2',
    #                loc=(0.02, 0.7), fontsize=8)
    #
    #     PF.add_standard_fig_info(fig)
    #     # endregion
    #
    # if show_plots['int_ent_others'] is True:
    #     fig, axs = PF.make_axes(3, single_fig_size=fig_size)
    #     # region dT adjusted s.t. integrated_data has dS = ln2, fit with that dt forced then integrated
    #     ax = axs[0]
    #     # PF.display_1d(x, int_avg_dt_ln2, ax, label='Averaged data')
    #     PF.display_1d(x, int_avg_dt_ln2, ax, label='int_avg_dt_ln2')
    #     # ax.plot(x_e_fit_avg, int_of_fit_dt_ln2, c='C3', label='integrated fit\nwith dT forced')
    #     ax.plot(x_e_fit_avg, int_of_fit_dt_ln2, c='C3', label='int_of_fit_dt_ln2')
    #     PF.ax_setup(ax, f'Dat[{dat.datnum}]:Integrated Entropy\ndT forced s.t. int_ds=Ln2', dat.Logs.x_label,
    #                 'Entropy /kB')
    #     _add_ln3_ln2(ax)
    #     _add_peak_final_text(ax, int_avg_dt_ln2, int_of_fit_dt_ln2)
    #     ax.legend(loc='lower right')
    #     PF.ax_text(ax, f'dT of forced fit={dt_ln2 / beta * 1000:.3f}mK\n'
    #                    f'amp = {i_fit_avg.best_values["amp"]:.3f}nA\n'
    #                    f'int_avg_dt_ln2 dS={int_avg_dt_ln2[-1] / np.log(2):.3f}kBLn2\n'
    #                    f'int_fit_dt_ln2 dS={int_of_fit_dt_ln2[-1] / np.log(2):.3f}kBLn2',
    #                loc=(0.02, 0.7), fontsize=8)
    #     # endregion
    #
    #     # region dT from Entropy fit, also integration of best fit
    #     ax = axs[1]
    #     # PF.display_1d(x, int_avg_dt_from_fit, ax, label='Averaged data')
    #     PF.display_1d(x, int_avg_dt_from_fit, ax, label='int_avg_dt_from_fit')
    #     # ax.plot(x_e_fit_avg, int_of_fit_dt_from_fit, c='C3', label='integrated fit\nwith dT from fit')
    #     ax.plot(x_e_fit_avg, int_of_fit_dt_from_fit, c='C3', label='int_of_fit_dt_from_fit')
    #     PF.ax_setup(ax, f'Dat[{dat.datnum}]:Integrated Entropy\ndT from entropy fit', dat.Logs.x_label, 'Entropy /kB')
    #     _add_ln3_ln2(ax)
    #     _add_peak_final_text(ax, int_avg_dt_from_fit, int_of_fit_dt_from_fit)
    #     ax.legend(loc='lower right')
    #     PF.ax_text(ax, f'dT = {dt_from_fit / beta * 1000:.3f}mK\n'
    #                    f'amp = {i_fit_avg.best_values["amp"]:.3f}nA\n'
    #                    f'int_avg_dt_from_fit dS=\n{int_avg_dt_from_fit[-1] / np.log(2):.3f}kBLn2\n'
    #                    f'int_of_fit_dt_fit dS=\n{int_of_fit_dt_from_fit[-1] / np.log(2):.3f}kBLn2',
    #                loc=(0.02, 0.6), fontsize=8)
    #     # endregion
    #
    #     # region dT from fit, amp s.t. int_fit dS = fit dS (scaling from fit)
    #     ax = axs[2]
    #     # PF.display_1d(x, int_avg_sf_from_fit, ax, label='Averaged data')
    #     PF.display_1d(x, int_avg_sf_from_fit, ax, label='int_avg_sf_from_fit')
    #     # ax.plot(x_e_fit_avg, int_of_fit_sf_from_fit, c='C3', label='integrated fit\nscaling from fit')
    #     ax.plot(x_e_fit_avg, int_of_fit_sf_from_fit, c='C3', label='int_of_fit_sf_from_fit')
    #     PF.ax_setup(ax,
    #                 f'Dat[{dat.datnum}]:Integrated Entropy\nscaling from fit (dT from fit\namp s.t. int_fit dS = fit dS)',
    #                 dat.Logs.x_label, 'Entropy /kB')
    #     ax.legend(loc='lower right')
    #     _add_ln3_ln2(ax)
    #     _add_peak_final_text(ax, int_avg_sf_from_fit, int_of_fit_sf_from_fit)
    #     PF.ax_text(ax, f'dT = {dt_from_fit / beta * 1000:.3f}mK\n'
    #                    f'amp = {amp_forced_fit_ds:.3f}nA\n'
    #                    f'int_avg_sf_from_fit dS=\n{int_avg_sf_from_fit[-1] / np.log(2):.3f}kBLn2\n'
    #                    f'int_of_fit_sf_from_fit dS=\n{int_of_fit_sf_from_fit[-1] / np.log(2):.3f}kBLn2',
    #                loc=(0.02, 0.6), fontsize=8)
    #     # endregion
    #     PF.add_standard_fig_info(fig)
    #
    # # endregion
