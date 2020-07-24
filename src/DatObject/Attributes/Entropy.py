import numpy as np
from typing import List, Union, Tuple
from src import HDF_Util as HDU
import src.CoreUtil as CU
from src.DatObject.Attributes import DatAttribute as DA
import src.Main_Config as cfg
import lmfit as lm
import pandas as pd
import h5py
import logging

logger = logging.getLogger(__name__)


def entropy_nik_shape(x, mid, theta, const, dS, dT):
    """fit to entropy curve"""
    arg = ((x - mid) / (2 * theta))
    return -dT * ((x - mid) / (2 * theta) - 0.5 * dS) * (np.cosh(arg)) ** (-2) + const


class NewEntropy(DA.FittingAttribute):
    version = '1.1'
    group_name = 'Entropy'

    """
    Versions:
        1.1 -- 20-7-20: Changed average_data to use centers not center_ids. Better way to average data
    """

    def __init__(self, hdf):
        self.angle = None  # type: Union[float, None]
        super().__init__(hdf)

    def get_from_HDF(self):
        super().get_from_HDF()  # Gets self.x/y/avg_fit/all_fits
        dg = self.group.get('Data', None)
        if dg is not None:
            self.data = dg.get('entropy_r', None)
            self.avg_data = dg.get('avg_entropy_r', None)
            self.avg_data_err = dg.get('avg_entropy_r_err', None)
        self.angle = self.group.attrs.get('angle', None)

    def update_HDF(self):
        super().update_HDF()
        self.group.attrs['angle'] = self.angle

    def recalculate_entr(self, centers, x_array=None):
        """
        Recalculate entropy r from 'entropy x' and 'entropy y' in HDF using center positions provided on x_array if
        provided otherwise on original x_array.

        Args:
            centers (np.ndarray):  Center positions in units of x_array (either original or passed)
            x_array (np.ndarray):  Option to pass an x_array that centers were defined on

        Returns:
            None: Sets self.data, self.angle, self.avg_data
        """
        x = x_array if x_array is not None else self.x
        dg = self.group['Data']
        entx = dg.get('entropy_x', None)
        enty = dg.get('entropy_y', None)
        assert entx not in [None, np.nan]
        if enty is None or enty.size == 1:
            entr = CU.center_data(x, entx, centers)  # To match entr which gets centered by calc_r
            angle = 0.0
        else:
            entr, angle = calc_r(entx, enty, x=x, centers=centers)
        self.data = entr
        self.angle = angle

        self.set_avg_data(centers='None')  # Because entr is already centered now
        self.update_HDF()

    def _set_data_hdf(self, **kwargs):
        super()._set_data_hdf(data_name='entropy_r')

    def run_row_fits(self, params=None, **kwargs):
        super().run_row_fits(entropy_fits, params=params)

    def _set_row_fits_hdf(self):
        super()._set_row_fits_hdf()

    def set_avg_data(self, centers=None, x_array=None):
        if centers is not None:
            logger.warning(f'Using centers to average entropy data, but data is likely already centered!')
        super().set_avg_data(centers=centers, x_array=x_array)  # sets self.avg_data/avg_data_err and saves to HDF

    def _set_avg_data_hdf(self):
        dg = self.group['Data']
        HDU.set_data(dg, 'avg_entropy_r', self.avg_data)
        HDU.set_data(dg, 'avg_entropy_r_err', self.avg_data_err)

    def run_avg_fit(self, params=None, **kwargs):
        super().run_avg_fit(entropy_fits, params=params)  # sets self.avg_fit and saves to HDF

    def _set_avg_fit_hdf(self):
        super()._set_avg_fit_hdf()

    def _set_default_group_attrs(self):
        super()._set_default_group_attrs()

    def _get_centers_from_transition(self):
        assert 'Transition' in self.hdf.keys()
        tg = self.hdf['Transition']  # type: h5py.Group
        rg = tg.get('Row fits', None)
        if rg is None:
            raise AttributeError("No Rows Group in self.hdf['Transition'], this must be initialized first")
        fit_infos = DA.rows_group_to_all_FitInfos(rg)
        x = self.x
        return CU.get_data_index(x, [fi.best_values.mid for fi in fit_infos])


def calc_r(entx, enty, x=None, centers=None):
    """
    Calculate R using constant phase determined at largest signal value of averaged data

    Args:
        entx (np.ndarray):  Entropy x signal (1D or 2D)
        enty (np.ndarray):  Entropy y signal (1D or 2D)
        x (np.ndarray): x_array for centering data with center values
        centers (np.ndarray): Center of transition to center data on

    Returns:
        (np.ndarray, float): 1D or 2D entropy r, phase angle
    """

    entx = np.atleast_2d(entx)
    enty = np.atleast_2d(enty)

    if x is None or centers is None:
        logger.warning('Not using centers to center data because x or centers missing')
        entxav = np.nanmean(entx, axis=0)
        entyav = np.nanmean(enty, axis=0)
    else:
        entxav = CU.mean_data(x, entx, centers, return_std=False)
        entyav = CU.mean_data(x, enty, centers, return_std=False)

    x_max, y_max, which = _get_max_and_sign_of_max(entxav, entyav)  # Gets max of x and y at same location
    # and which was bigger
    angle = np.arctan(y_max / x_max)

    entr = np.array([x * np.cos(angle) + y * np.sin(angle) for x, y in zip(entx, enty)])
    entangle = angle

    if entr.shape[0] == 1:  # Return to 1D if only one row of data
        entr = np.squeeze(entr, axis=0)
    return entr, entangle


def get_param_estimates(x_array, data, mids=None, thetas=None) -> List[lm.Parameters]:
    if data.ndim == 1:
        return [_get_param_estimates_1d(x_array, data, mids, thetas)]
    elif data.ndim == 2:
        mids = mids if mids is not None else [None] * data.shape[0]
        thetas = thetas if thetas is not None else [None] * data.shape[0]
        return [_get_param_estimates_1d(x_array, z, mid, theta) for z, mid, theta in zip(data, mids, thetas)]


def _get_param_estimates_1d(x, z, mid=None, theta=None) -> lm.Parameters:
    """Returns estimate of params and some reasonable limits. Const forced to zero!!"""
    params = lm.Parameters()
    dT = np.nanmax(z) - np.nanmin(z)
    if mid is None:
        mid = (x[np.nanargmax(z)] + x[np.nanargmin(z)]) / 2  #
    if theta is None:
        theta = abs((x[np.nanargmax(z)] - x[np.nanargmin(z)]) / 2.5)

    params.add_many(('mid', mid, True, None, None, None, None),
                    ('theta', theta, True, 0, 500, None, None),
                    ('const', 0, False, None, None, None, None),
                    ('dS', 0, True, -5, 5, None, None),
                    ('dT', dT, True, -10, 50, None, None))

    return params





def entropy_1d(x, z, params: lm.Parameters = None, auto_bin=False):
    entropy_model = lm.Model(entropy_nik_shape)
    z = pd.Series(z, dtype=np.float32)
    if np.count_nonzero(~np.isnan(z)) > 10:  # Don't try fit with not enough data
        z, x = CU.remove_nans(z, x)
        if auto_bin is True and len(z) > cfg.FIT_NUM_BINS:
            logger.debug(f'Binning data of len {len(z)} before fitting')
            bin_size = int(np.ceil(len(z) / cfg.FIT_NUM_BINS))
            x, z = CU.bin_data([x, z], bin_size)
        if params is None:
            params = get_param_estimates(x, z)[0]

        result = entropy_model.fit(z, x=x, params=params, nan_policy='omit')
        return result
    else:
        return None


def entropy_fits(x, z, params: List[lm.Parameters] = None, auto_bin=False):
    if params is None:
        params = [None] * z.shape[0]
    else:
        params = CU.ensure_params_list(params, z)
    if z.ndim == 1:  # 1D data
        return [entropy_1d(x, z, params[0], auto_bin=auto_bin)]
    elif z.ndim == 2:  # 2D data
        fit_result_list = []
        for i in range(z.shape[0]):
            fit_result_list.append(entropy_1d(x, z[i, :], params[i], auto_bin=auto_bin))
        return fit_result_list


def _get_max_and_sign_of_max(x, y) -> Tuple[float, float, np.array]:
    """Returns value of x, y at the max position of the larger of the two and which was larger...
     i.e. x and y value at index=10 if max([x,y]) is x at x[10] and 'x' because x was larger

    Args:
        x (np.ndarray): x data (can be nD but probably better to average first to use 1D)
        y (np.ndarray): y data (can be nD but probably better to average first to use 1D)

    Returns:
        (float, float, str): x_max, y_max, which was larger of 'x' and 'y'
    """

    if np.nanmax(np.abs(x)) > np.nanmax(np.abs(y)):
        which = 'x'
        x_max, y_max = _get_values_at_max(x, y)
    else:
        which = 'y'
        y_max, x_max = _get_values_at_max(y, x)
    return x_max, y_max, which


def _get_values_at_max(larger, smaller) -> Tuple[float, float]:
    """Returns values of larger and smaller at position of max in larger

    Useful for calculating phase difference between x and y entropy data. Best to do at
    place where there is a large signal and then hold constant over the rest of the data

    Args:
        larger (np.ndarray): Data with the largest abs value
        smaller (np.ndarray): Data with the smaller abs value to be evaluated at the same index as the larger data

    Returns:
        (float, float): max(abs) of larger, smaller at same index
    """
    assert larger.shape == smaller.shape
    if np.abs(np.nanmax(larger)) > np.abs(np.nanmin(larger)):
        large_max = np.nanmax(larger)
        index = np.nanargmax(larger)
    else:
        large_max = float(np.nanmin(larger))
        index = np.nanargmin(larger)
    small_max = smaller[index]
    return large_max, small_max


def integrate_entropy(data, scaling):
    """Integrates entropy data with scaling factor along last axis

    Args:
        data (np.ndarray): Entropy data
        scaling (float): scaling factor from dT, amplitude, dx

    Returns:
        np.ndarray: Integrated entropy units of Kb with same shape as original array
    """

    return np.nancumsum(data, axis=-1) * scaling


def scaling(dt, amplitude, dx):
    """Calculate scaling factor for integrated entropy from dt, amplitude, dx

    Args:
        dt (float): The difference in theta of hot and cold (in units of plunger gate).
            Note: Using lock-in dT is 1/2 the peak to peak, for Square wave it is the full dT
        amplitude (float): The amplitude of charge transition from the CS
        dx (float): How big the DAC steps are in units of plunger gate
            Note: Relative to the data passed in, not necessarily the original x_array

    Returns:
        float: Scaling factor to multiply cumulative sum of data by to conver to entropy
    """
    return dx / amplitude / dt




# def plot_standard_entropy(dat, axs, plots: List[int] = (1, 2, 3), kwargs_list: List[dict] = None):
#     """This returns a list of axes which show normal useful entropy plots (assuming 2D for now)
#     It requires a dat object to be passed to it so it has access to all other info
#     1. 2D entr (or entx if no enty)
#     2. Centered and averaged 1D entropy
#     3. 1D slice of entropy R
#     4. Nik_Entropy per repeat
#     5. 2D entx
#     6. 2D enty
#     7. 1D slice of entx
#     8. 1D slice of enty
#     9. Integrated entropy
#     10. Integrated entropy per line
#     11. Add DAC table and other info
#
#     Kwarg hints:
#     swap_ax:bool, swap_ax_labels:bool, ax_text:bool"""
#
#     Entropy = dat.Entropy
#     Data = dat.Data
#
#     assert len(axs) >= len(plots)
#     if kwargs_list is not None:
#         assert len(kwargs_list) == len(plots)
#         assert type(kwargs_list[0]) == dict
#         kwargs_list = [{**k, 'no_datnum': True} if 'no_datnum' not in k.keys() else k for k in kwargs_list]  # Make
#         # no_datnum default to True if not passed in.
#     else:
#         kwargs_list = [{'no_datnum': True}] * len(plots)
#
#     i = 0
#
#     if 1 in plots:  # Add 2D entr (or entx)
#         ax = axs[i]
#         ax.cla()
#         if dat.Entropy.entr is not None:
#             data = Entropy.entr
#             title = 'Entropy R'
#         elif dat.Entropy.entx is not None:
#             data = Entropy.entx
#             title = 'Entropy X'
#         else:
#             raise AttributeError(f'No entr or entx for dat{dat.datnum}[{dat.datname}]')
#         ax = PF.display_2d(Data.x_array, Data.y_array, data, ax, x_label=dat.Logs.x_label,
#                            y_label=dat.Logs.y_label, dat=dat, title=title, **kwargs_list[i])
#
#         axs[i] = ax
#         i += 1  # Ready for next plot to add
#
#     if 2 in plots:  # Add Centered and Averaged 1D entropy
#         ax = axs[i]
#         ax.cla()
#         if Entropy.entrav is not None:
#             data = Entropy.entrav
#             title = 'Avg Entropy R'
#         elif Entropy.entxav is not None:
#             data = Entropy.entxav
#             title = 'Avg Entropy X'
#         else:
#             raise AttributeError(f'No entrav or entxav for dat{dat.datnum}[{dat.datname}]')
#         fit = dat.Entropy._avg_full_fit
#         ax = PF.display_1d(Data.x_array - np.average(Data.x_array), data, ax=ax, x_label=f'Centered {dat.Logs.x_label}',
#                            y_label='1D Avg Entropy Signal', dat=dat, title=title, scatter=True, **kwargs_list[i])
#         ax.plot(dat.Entropy.avg_x_array-np.average(Data.x_array), fit.best_fit, color='C3')
#         PF.ax_text(ax, f'dS={Entropy.avg_fit_values.dSs[0]:.3f}')
#         axs[i] = ax
#         i += 1
#
#     if 3 in plots:  # Add 1D entr
#         ax = axs[i]
#         ax.cla()
#         if dat.Entropy.entr is not None:
#             data = Entropy.entr[round(dat.Entropy.entr.shape[0] / 2)]
#         else:
#             raise AttributeError(f'No entyav for dat{dat.datnum}[{dat.datname}]')
#         ax = PF.display_1d(Data.x_array, data, ax, x_label=dat.Logs.x_label,
#                            y_label='Entropy signal', dat=dat, title='1D entropy R', **kwargs_list[i])
#         axs[i] = ax
#         i += 1  # Ready for next plot to add
#
#     if 4 in plots:  # Add Nik_Entropy per repeat
#         ax = axs[i]
#         ax.cla()
#         if Entropy.fit_values is not None:
#             data = Entropy.fit_values.dSs
#             dataerr = [param['dS'].stderr for param in Entropy.params]
#         else:
#             raise AttributeError(f'No Entropy.fit_values for {dat.datnum}[{dat.datname}]')
#         ax = PF.display_1d(Data.y_array, data, errors=dataerr, ax=ax, x_label='Entropy/kB', y_label=dat.Logs.y_label,
#                            dat=dat, title='Nik Entropy', swap_ax=True, **kwargs_list[i])
#         if kwargs_list[i].get('swap_ax', False) is False:
#             ax.axvline(np.log(2), c='k', ls=':')
#         else:
#             ax.axhline(np.log(2), c='k', ls=':')
#
#         axs[i] = ax
#         i += 1
#
#     if 5 in plots:  # Add 2D entx
#         ax = axs[i]
#         ax.cla()
#         if dat.Entropy.entx is not None:
#             data = Entropy.entx
#         else:
#             raise AttributeError(f'No entx for dat{dat.datnum}[{dat.datname}]')
#         ax = PF.display_2d(Data.x_array, Data.y_array, data, ax, x_label=dat.Logs.x_label,
#                            y_label=dat.Logs.y_label, dat=dat, title='Entropy X', **kwargs_list[i])
#         axs[i] = ax
#         i += 1  # Ready for next plot to add
#
#     if 6 in plots:  # Add 2D enty
#         ax = axs[i]
#         ax.cla()
#         if dat.Entropy.entx is not None:
#             data = Entropy.entx
#         else:
#             raise AttributeError(f'No entx for dat{dat.datnum}[{dat.datname}]')
#         ax = PF.display_2d(Data.x_array, Data.y_array, data, ax, x_label=dat.Logs.x_label,
#                            y_label=dat.Logs.y_label, dat=dat, title='Entropy y', **kwargs_list[i])
#         axs[i] = ax
#         i += 1  # Ready for next plot to add
#
#     if 7 in plots:  # Add 1D entx
#         ax = axs[i]
#         ax.cla()
#         if dat.Entropy.entx is not None:
#             data = Entropy.entx[0]
#         else:
#             raise AttributeError(f'No entxav for dat{dat.datnum}[{dat.datname}]')
#         ax = PF.display_1d(Data.x_array, data, ax, x_label=dat.Logs.x_label,
#                            y_label='Entropy signal', dat=dat, title='1D entropy X', **kwargs_list[i])
#         axs[i] = ax
#         i += 1  # Ready for next plot to add
#
#     if 8 in plots:  # Add 1D entx
#         ax = axs[i]
#         ax.cla()
#         if dat.Entropy.enty is not None:
#             data = Entropy.enty[0]
#         else:
#             raise AttributeError(f'No entyav for dat{dat.datnum}[{dat.datname}]')
#         ax = PF.display_1d(Data.x_array, data, ax, x_label=dat.Logs.x_label,
#                            y_label='Entropy signal', dat=dat, title='1D entropy Y', **kwargs_list[i])
#         axs[i] = ax
#         i += 1  # Ready for next plot to add
#
#     if 9 in plots:  # Add integrated entropy
#         ax = axs[i]
#         ax.cla()
#         k = kwargs_list[i]
#         if 'ax_text' not in k.keys():  # Default to ax_text == True
#             k = {**k, 'ax_text': True}
#         if 'loc' not in k.keys():
#             k = {**k, 'loc': (0.1, 0.5)}
#
#         if dat.Entropy.int_entropy_initialized is True:
#             data = dat.Entropy.integrated_entropy
#             x = dat.Entropy.integrated_entropy_x_array
#             PF.display_1d(x, data, ax, y_label='Entropy /kB', dat=dat, **k)
#             if dat.Entropy.int_ds > 0:
#                 expected = np.log(2)
#             else:
#                 expected = -np.log(2)
#             ax.axhline(expected, c='k', ls=':')
#             err = dat.Entropy.scaling_err
#             ax.fill_between(x, data * (1 - err), data * (1 + err), color='#AAAAAA')
#             if k['ax_text'] is True:
#                 PF.ax_text(ax, f'dS = {dat.Entropy.int_ds:.3f}\n'
#                            f'SF = {dat.Entropy.scaling:.4f}\n'
#                            f'SFerr = {dat.Entropy.scaling_err*100:.0f}%\n'
#                             f'dT = {dat.Entropy._int_dt:.3f}mV\n'
#                                f'amp = {dat.Entropy._amp:.2f}nA',
#                            loc=(k['loc']))
#
#
#         else:
#             print('Need to initialize integrated entropy first')
#         axs[i] = ax
#         i += 1
#
#     if 10 in plots:  # Add integrated entropy per line
#         ax = axs[i]
#         ax.cla()
#         k = kwargs_list[i]
#         if 'ax_text' not in k.keys():  # Default to ax_text == True
#             k = {**k, 'ax_text': True}
#         if 'loc' not in k.keys():
#             k = {**k, 'loc': (0.1, 0.5)}
#
#         if dat.Entropy.int_entropy_initialized is True:
#             x = dat.Entropy.integrated_entropy_x_array
#             data = dat.Entropy.int_entropy_per_line
#             _plot_integrated_entropy_per_line(ax, x, data, x_label=dat.Logs.x_label, y_label='Entropy/kB',
#                                               title='Integrated Entropy')
#             if dat.Entropy.int_ds > 0:
#                 expected = np.log(2)
#             else:
#                 expected = -np.log(2)
#             # ax.axhline(expected, c='k', ls=':')
#             err = dat.Entropy.scaling_err
#             avg_data = dat.Entropy.integrated_entropy
#             ax.fill_between(x, avg_data * (1 - err), avg_data * (1 + err), color='#AAAAAA')
#             if k['ax_text'] is True:
#                 PF.ax_text(ax, f'dS = {dat.Entropy.int_ds:.3f}\n'
#                            f'SF = {dat.Entropy.scaling:.4f}\n'
#                            f'SFerr = {dat.Entropy.scaling_err*100:.0f}%\n'
#                             f'dT = {dat.Entropy._int_dt:.3f}mV\n'
#                                f'amp = {dat.Entropy._amp:.2f}nA',
#                            loc=(k['loc']))
#
#         else:
#             print('Need to initialize integrated entropy first')
#         axs[i] = ax
#         i += 1
#
#     if 11 in plots:
#         ax = axs[i]
#         PF.plot_dac_table(ax, dat)
#         fig = plt.gcf()
#         try:
#             fig.suptitle(f'Dat{dat.datnum}')
#             PF.add_standard_fig_info(fig)
#             if dat.Logs.sweeprate is not None:
#                 sr = f'{dat.Logs.sweeprate:.0f}mV/s'
#             else:
#                 sr = 'N/A'
#             PF.add_to_fig_text(fig,
#                            f'ACbias = {dat.Instruments.srs1.out / 50 * np.sqrt(2):.1f}nA, sweeprate={sr}, temp = {dat.Logs.temp:.0f}mK')
#         except AttributeError:
#             print(f'One of the attributes was missing for dat{dat.datnum} so extra fig text was skipped')
#         axs[i] = ax
#         i+=1
#
#     return axs

#
# def recalculate_entropy_with_offset_subtracted(dat, update=True, save=True, dfname='default'):
#     """takes the params for the current fits, changes the const to be allowed to vary, fits again, subtracts that
#     offset from each line of data, then fits again. Does NOT recalculate integrated entropy"""
#     from src.DFcode.DatDF import update_save
#     if dat.Entropy.avg_params['const'].vary is False:
#         params = dat.Entropy.params
#         for p in params:
#             p['const'].vary = True
#         dat.Entropy.recalculate_fits(params)
#         assert dat.Entropy.avg_params['const'].vary is True
#
#     dat.Entropy._data = np.array(
#         [data - c for data, c in zip(dat.Entropy._data, dat.Entropy.fit_values.consts)]).astype(np.float32)
#     dat.datname = 'const_subtracted_entropy'
#     dat.Entropy.recalculate_fits()
#     update_save(dat, update, save, dfname=dfname)
#
#
# def recalculate_int_entropy_with_offset_subtracted(dat, dc=None, dT_mV=None, make_new=False, update=True,
#                                                    save=True, datdf=None):
#     """
#     Recalculates entropy with offset subtracted, then recalculates integrated entropy with offset subtracted
#
#     @param dc: dcdat object to use for calculating dT_mV for integrated fit. Otherwise can pass in dT_mV value
#     @type dc: Dat
#     @param make_new: Saves dat with name 'const_subtracted_entropy' otherwise will just overwrite given instance
#     @type make_new: bool
#     @param update: Whether to update the DF given
#     @type update: bool
#     @param dfname: Name of dataframe to update changes in
#     @type dfname: str
#     @return: None
#     @rtype: None
#     """
#     from src.DFcode.DatDF import update_save
#     datname = dat.datname
#     if datname != 'const_subtracted_entropy' and make_new is False:
#         ans = CU.option_input(
#             f'datname=[{dat.datname}], do you want to y: create a new copy with entropy subtracted, n: change this '
#             f'copy, a: abort?',
#             {'y': True, 'n': False, 'a': 'abort'})
#         if ans == 'abort':
#             return None
#         elif ans is True:
#             make_new = True
#         elif ans is False:
#             make_new = False
#         else:
#             raise NotImplementedError
#
#     recalculate_entropy_with_offset_subtracted(dat, update=False, save=False)  # Creates dat with name changed
#     if make_new is True:
#         pass
#     elif make_new is False:
#         dat.datname = datname  # change name back to original before any saving or updating
#     else:
#         raise NotImplementedError
#     if dT_mV is not None:
#         dt = dT_mV
#         dat.Entropy.init_integrated_entropy_average(dT_mV=dt, dT_err=0,
#                                                     amplitude=dat.Transition.avg_fit_values.amps[0],
#                                                     amplitude_err=0)
#     elif dc is not None:
#         dt = dc.DCbias.get_dt_at_current(dat.Instruments.srs1.out / 50 * np.sqrt(2))
#         dat.Entropy.init_integrated_entropy_average(dT_mV=dt, dT_err=0,
#                                                     amplitude=dat.Transition.avg_fit_values.amps[0],
#                                                     amplitude_err=0, dcdat=dc)
#     else:
#         print('ERROR[E.recalculate_int_entropy_with_offset_corrected]: Must provide either "dT_mV" or "dc" to '
#               'calculate integrated entropy.\r Entropy has been recalculated with offset removed, but nothing has been '
#               'saved to DF')
#         return None
#
#     if datdf is not None:
#         update_save(dat, update, save, datdf=datdf)
#     elif save is True or update is True:
#         print('WARNING[_recalculate_int_entropy_with_offset_subtracted]: No datdf provided to carry out update or save')
#
#
# def plot_entropy_along_transition(dats, fig=None, axs=None, x_axis='gamma', exclude=None):
#     """
#     For plotting dats along a transition. I.e. each dat is a repeat measurement somewhere along transition
#
#     @param exclude: datnums to exclude from plot
#     @type exclude: List[int]
#     @param dats: list of dat objects
#     @type dats: src.DatObject.Dat.Dat
#     @param fig:
#     @type fig: plt.Figure
#     @param axs:
#     @type axs: List[plt.Axes]
#     @return:
#     @rtype: plt.Figure, List[plt.Axes]
#     """
#
#     if exclude is not None:
#         dats = [dat for dat in dats if dat.datnum not in exclude]  # remove excluded dats from plotting
#
#     if axs is None:
#         fig, axs = PF.make_axes(3)
#
#     PF.add_standard_fig_info(fig)
#
#     if x_axis.lower() == 'rct':
#         xs = [dat.Logs.fdacs[4] for dat in dats]
#     elif x_axis.lower() == 'rcss':
#         xs = [dat.Logs.fdacs[6] for dat in dats]
#     elif x_axis.lower() == 'gamma':
#         xs = [dat.Transition.avg_fit_values.gs[0] for dat in dats]
#     elif x_axis.lower() == 'mar_sdr':
#         xs = [dat.Logs.dacs[13] for dat in dats]
#     else:
#         print('x_axis has to be one of [rct, gamma, rcss, mar_sdr]')
#
#     ax = axs[0]
#     PF.ax_setup(ax, title=f'Nik Entropy vs {x_axis}', x_label=f'{x_axis} /mV', y_label='Entropy /kB', legend=False, fs=10)
#     for dat, x in zip(dats, xs):
#         y = dat.Entropy.avg_fit_values.dSs[0]
#         yerr = np.std(dat.Entropy.fit_values.dSs)
#         ax.errorbar(x, y, yerr=yerr, linestyle=None, marker='x')
#
#     ax = axs[1]
#     PF.ax_setup(ax, title=f'Integrated Entropy vs {x_axis}', x_label=f'{x_axis} /mV', y_label='Entropy /kB', legend=False,
#              fs=10)
#     for dat, x in zip(dats, xs):
#         y = dat.Entropy.int_ds
#         yerr = np.std(dat.Entropy.int_entropy_per_line[-1])
#         ax.errorbar(x, y, yerr=yerr, linestyle=None, marker='x')
#
#     ax = axs[2]
#     for dat in dats:
#         x = dat.Entropy.x_array - dat.Transition.mid
#         ax.plot(x, dat.Entropy.integrated_entropy, linewidth=1)
#     PF.ax_setup(ax, title=f'Integrated Entropy vs {x_axis}', x_label=dats[0].Logs.x_label, y_label='Entropy /kB', fs=10)
#
#     plt.tight_layout(rect=(0, 0.1, 1, 1))
#     PF.add_standard_fig_info(fig)
#     return fig, axs


