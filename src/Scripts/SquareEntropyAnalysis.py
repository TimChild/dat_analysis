from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, TYPE_CHECKING, Tuple, Callable
import numpy as np
from src.CoreUtil import bin_data
import src.CoreUtil as CU
from src.DatObject.Attributes.SquareEntropy import average_2D, entropy_signal, scaling, IntegratedInfo, \
    integrate_entropy
from src.DatObject.Attributes.Transition import transition_fits, i_sense
from src.DatObject.Attributes import Entropy as E, DatAttribute as DA
import re
import lmfit as lm
import logging
import pandas as pd
from collections.abc import Iterable, Sized

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF


def merge_dat_parts(dats, centers=None):  # TODO: This could be a lot more general and not only work for SE
    """
    Merges two part Square entropy dats.
    Args:
        dats (Tuple[DatHDF, DatHDF]):
        centers (Optional[np.ndarray]):

    Returns:
        Tuple[np.ndarray, np.ndarray]: full_x, full_cycled_data
    """
    assert len(dats) == 2
    p1_dat, p2_dat = None, None
    for dat in dats:
        part_num = dat.Logs.part_of[0]
        if part_num == 1:
            p1_dat = dat
        elif part_num == 2:
            p2_dat = dat

    assert None not in [p1_dat, p2_dat]

    full_x = p1_dat.Data.x_array
    p1_data = p1_dat.SquareEntropy.Processed.outputs.cycled
    full_x = CU.get_matching_x(full_x, p1_data)

    p2_x = p2_dat.Data.x_array
    p2_data = p2_dat.SquareEntropy.Processed.outputs.cycled
    p2_x = CU.get_matching_x(p2_x, p2_data)
    idx = CU.get_data_index(full_x, p2_x[0])

    # TODO: make this better
    if p1_data.ndim == 2:
        p1_data = p1_data.reshape((1, *p1_data.shape))
    if p2_data.ndim == 2:
        p2_data = p2_data.reshape((1, *p2_data.shape))

    p2_data = np.pad(p2_data, ((0, 0), (0, 0), (idx, full_x.shape[0] - p2_x.shape[0] - idx)), mode='constant',
                     constant_values=np.nan)

    full_data = np.concatenate((p1_data, p2_data), axis=0)
    full_x, full_avg_data = average_2D(full_x, full_data, avg_nans=True, centers=centers)
    return full_x, full_avg_data


@dataclass
class EA_params:
    """
    Parameters for carrying out Square Entropy processing on arbitrary data.
    Note: ...fit_ranges are in units of EA_data.x not index positions. None will mean beginning and end
    """
    bin_data: bool
    num_per_row: int

    sub_line: bool
    sub_line_range: Tuple[float, float]

    int_entropy_range: Tuple[float, float]

    allowed_amp_range: Tuple[float, float]
    default_amp: float
    allowed_dT_range: Tuple[float, float]
    default_dT: float

    CT_fit_range: Tuple[Optional[float], Optional[float]] = field(
        default=(None, None))  # Useful if fitting with two part dat

    fit_param_edit_kwargs: dict = field(default_factory=dict)
    E_fit_range: Tuple[Optional[float], Optional[float]] = field(
        default=(None, None))  # Useful if fitting with two part dat or avoiding bumps


@dataclass
class EA_datas:
    xs: List[np.ndarray] = field(default_factory=list)
    trans_datas: List[np.ndarray] = field(default_factory=list)
    entropy_datas: List[np.ndarray] = field(default_factory=list)
    integrated_datas: List[np.ndarray] = field(default_factory=list)
    data_minus_fits: List[np.ndarray] = field(default_factory=list)

    def __len__(self):
        len_x = len(self.xs)
        len_td = len(self.trans_datas)
        len_ed = len(self.entropy_datas)
        len_id = len(self.integrated_datas)
        len_dmf = len(self.data_minus_fits)
        return int(np.nanmax([len_x, len_td, len_ed, len_id, len_dmf]))

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError
        data = EA_data()
        if item < len(self.xs):
            data.x = self.xs[item]
        if item < len(self.trans_datas):
            data.trans_data = self.trans_datas[item]
        if item < len(self.entropy_datas):
            data.entropy_data = self.entropy_datas[item]
        if item < len(self.integrated_datas):
            data.integrated_data = self.integrated_datas[item]
        if item < len(self.data_minus_fits):
            data.data_minus_fit = self.data_minus_fits[item]
        return data

    @classmethod
    def from_dats(cls, dats: List[DatHDF]):
        datas = cls()
        for dat in dats:
            if dat.Logs.part_of is not None and dat.Logs.part_of[0] == 2:  # By default use part1 of dats
                continue
            data = get_data(dat)
            if data is None:
                logger.warning(f'Dat{dat.datnum} had no EA_data in dat.Other')
                continue
            datas.xs.append(data.x)
            datas.trans_datas.append(data.trans_data)
            datas.entropy_datas.append(data.entropy_data)
            datas.integrated_datas.append(data.integrated_data)
            datas.data_minus_fits.append(data.data_minus_fit)
        return datas

    @classmethod
    def from_dat(cls, dat: DatHDF):
        datas = cls()
        for name in datas.__annotations__.keys():
            i = 0
            data_list = getattr(datas, name)
            while True:
                k = name + str(i)
                if k in dat.Other.Data.keys():
                    data_list.append(dat.Other.Data.get(k))
                    i += 1
                else:
                    break
        return datas

    def add_fit_to_entropys(self, fits: List[DA.FitInfo]):  # Note: I don't think this can be saved in the HDF
        self.entropy_datas = [[fit.eval_fit(x), edata] for fit, x, edata in zip(fits, self.xs, self.entropy_datas)]

    def append(self, data: EA_data):
        self.xs.append(data.x)
        self.trans_datas.append(data.trans_data)
        self.entropy_datas.append(data.entropy_data)
        self.integrated_datas.append(data.integrated_data)
        self.data_minus_fits.append(data.data_minus_fit)


@dataclass
class EA_data:
    x: np.ndarray = field(default=None)
    trans_data: np.ndarray = field(default=None)
    entropy_data: np.ndarray = field(default=None)
    integrated_data: np.ndarray = field(default=None)
    data_minus_fit: np.ndarray = field(default=None)


@dataclass
class EA_titles:
    ids: list = field(default_factory=list)
    trans: list = field(default_factory=list)
    entropy: list = field(default_factory=list)
    integrated: list = field(default_factory=list)
    data_minus_fit: list = field(default_factory=list)


def data_from_dat_pair(dat_pair: Tuple[DatHDF, DatHDF], centers=None) -> EA_data:
    """
    Takes dat pair and combines them into 1D averaged dataset, and puts the data in EA_Data class
    Args:
        dat_pair (Tuple[DatHDF, DatHDF]): Both two part dats in a Tuple/List
        centers (Optional[Union[list, np.ndarray]): Centers of transitions to use for averaged data together (otherwise
            will automatically try to center with i_sense fit if LCT > -350

    Returns:
        (EA_Data): 1D Data class with x, transition, entropy, and space for integrated entropy data.
    """
    dat = dat_pair[0]
    if centers is None:
        if dat.Logs.fds["LCT"] > -350:
            centers = [0] * len(dat.Data.y_array)

    x, averaged_data = merge_dat_parts(dat_pair, centers=centers)  # centers = None finds center with transition fit
    entropy_data = entropy_signal(averaged_data)
    data = EA_data(x=x, trans_data=averaged_data, entropy_data=entropy_data)
    return data


@dataclass
class EA_value:
    sf: float = field(default=None)
    tc: float = field(default=None)
    th: float = field(default=None)
    dT: float = field(default=None)
    amp: float = field(default=None)
    mid: float = field(default=None)
    dx: float = field(default=None)
    int_dS: float = field(default=None)
    fit_dS: float = field(default=None)  # TODO: populate this properly
    efit_info: DA.FitInfo = field(default=None)
    data_minus_fit: float = field(default=None)

    @property
    def dS(self):
        if self.efit_info:
            dS = CU.get_nested_attr_default(self.efit_info, 'best_values.dS', None)
            return dS
        else:
            return None


@dataclass
class EA_values:
    sfs: List[float] = field(default_factory=list)
    tcs: List[float] = field(default_factory=list)
    ths: List[float] = field(default_factory=list)
    dTs: List[float] = field(default_factory=list)
    amps: List[float] = field(default_factory=list)
    mids: List[float] = field(default_factory=list)
    dxs: List[float] = field(default_factory=list)
    int_dSs: List[float] = field(default_factory=list)
    fit_dSs: List[float] = field(default_factory=list)  # TODO: populate this properly
    efit_infos: List[DA.FitInfo] = field(default_factory=list)
    data_minus_fits: list[float] = field(default_factory=list)

    def __len__(self):
        lens = [len(getattr(self, k, [])) for k in self.__annotations__.keys()]
        return int(np.nanmax(lens))

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError
        value = EA_value()
        for k in self.__annotations__.keys():
            vs = getattr(self, k, [])
            if item < len(vs):
                setattr(value, k[:-1], vs[item])

        return value

    def append(self, value: EA_value):
        for k in self.__annotations__.keys():
            list_ = getattr(self, k)
            list_.append(getattr(value, k[:-1]))

    @property
    def dSs(self):
        if self.efit_infos:
            dSs = [CU.get_nested_attr_default(f, 'best_values.dS', None) for f in self.efit_infos]
            return dSs
        else:
            return None

    # @property
    # def fit_dSs(self):  # TODO: This is a temporary fix to not having fit_dSs popluated above... might turn into permanent fix...
    #     return self.dSs
    #
    # @fit_dSs.setter
    # def fit_dSs(self, value):
    #     pass

    @classmethod
    def from_dats(cls, dats: List[DatHDF], uncertainty=False):
        """
        Gets EA_value from each dat and combines into EA_values class
        Args:
            dats (List[DatHDF]):
            uncertainty (bool): Whether to look for uncertainties instead of values (stored in similar way)

        Returns:
            EA_values: All values from dats combined
        """
        values = cls()
        if uncertainty:
            dat_key = 'EA_uncertainties'
        else:
            dat_key = 'EA_values'
        for dat in dats:
            if dat.Logs.part_of is not None and dat.Logs.part_of[0] == 2:  # By default use part1 of dats
                continue
            value = getattr(dat.Other, dat_key, None)
            if value is None:
                logger.warning(f'Dat{dat.datnum} had no {dat_key} in dat.Other')
                continue
            for k in cls.__annotations__.keys():
                list_ = getattr(values, k)
                list_.append(getattr(value, k[:-1], None))
        return values

    @classmethod
    def from_dat(cls, dat: DatHDF):
        values = getattr(dat.Other, 'EA_values', None)
        class_name = values.__class__.__name__
        if values is None:
            logger.warning(f'No dat.Other.EA_values found for dat{dat.datnum}')
            return None
        elif class_name == 'EA_value':
            logger.warning(f'Dat{dat.datnum} has single set of EA_value only, not EA_values')
            return None
        elif class_name == 'EA_values':
            efits = list()
            i = 0
            while True:
                name = f'efit_info_{i}'
                if hasattr(dat.Other, name):
                    efits.append(getattr(dat.Other, name))
                    i += 1
                else:
                    break
            values.efit_infos = efits  # Should also set it in the dat I think?
            return values


@dataclass
class DataCTvalues:
    tcs: List[float] = field(default_factory=list)
    ths: List[float] = field(default_factory=list)
    amps: List[float] = field(default_factory=list)
    mids: List[float] = field(default_factory=list)


def calculate_CT_values(data: EA_data, values: EA_value, params: EA_params) -> DataCTvalues:
    """
    Calculates the charge transition fit stuff, stores the interesting final values in 'values', but
    also returns ct_values which has each individual fit value used
    Args:
        data (EA_data): 1D EA data class
        values (EA_value): 1D values class
        params (EA_params): Params for analysis

    Returns:
        (DataCTvalues): Each fit value used to fill 'values', mostly for debugging
    """
    indexs = CU.get_data_index(data.x, params.CT_fit_range)
    t_fit_data = data.trans_data[:, indexs[0]:indexs[1]]
    x = data.x[indexs[0]:indexs[1]]
    ct_values = DataCTvalues()
    for data in t_fit_data[0::2]:
        fit = transition_fits(x, data, func=i_sense)[0]
        ct_values.tcs.append(fit.best_values['theta'])
        ct_values.amps.append(fit.best_values['amp'])
        ct_values.mids.append(fit.best_values['mid'])
    for data in t_fit_data[1::2]:
        fit = transition_fits(x, data, func=i_sense)[0]
        ct_values.ths.append(fit.best_values['theta'])

    values.tc = np.nanmean(ct_values.tcs)
    values.th = np.nanmean(ct_values.ths)

    mid = np.nanmean(ct_values.mids)
    amp = np.nanmean(ct_values.amps)
    dT = values.th - values.tc

    if -1000 < mid < 1000:
        values.mid = mid
    else:
        logger.warning(f'Mid = {mid:.2f}: Using 0 instead')
        values.mid = 0

    if params.allowed_amp_range[0] < amp < params.allowed_amp_range[1]:
        values.amp = amp
    else:
        logger.info(f'Amp={amp:.2f}: Default of {params.default_amp:.2f} Used instead')
        values.amp = params.default_amp
    if params.allowed_dT_range[0] < dT < params.allowed_dT_range[1]:
        values.dT = dT
    else:
        logger.info(f'dT={dT:.2f}: Default of {params.default_dT:.2f} Used instead')
        values.dT = params.default_dT
    return ct_values


def calculate_integrated(data: EA_data, values: EA_value, params: EA_params):
    """
    Integrates data.entropy_data and stores in data.integrated_data. Will sub line or not based on params
    Args:
        data (EA_data): 1D EA data class
        values (EA_value): 1D values class
        params (EA_params): Params for analysis

    Returns:
        None: Just adds data.integrated_data, and values.[dx, sf, int_dS]
    """
    values.dx = float(np.mean(np.diff(data.x)))
    values.sf = scaling(dt=values.dT, amplitude=values.amp, dx=values.dx)
    integrated_data = integrate_entropy(data.entropy_data, values.sf)

    if params.sub_line:
        line = lm.models.LinearModel()
        indexs = CU.get_data_index(data.x,
                                   [values.mid + params.sub_line_range[0], values.mid + params.sub_line_range[1]])
        if indexs[0] == indexs[1]:
            indexs = [0, int(round(data.x.shape[-1] / 20))]
            logger.warning(
                'Indexs for subtracting line from Integrated entropy were equal so defaulting to first 5% of data')
        i_data = integrated_data[indexs[0]:indexs[1]]
        line_fit = line.fit(i_data, x=data.x[indexs[0]:indexs[1]], nan_policy='omit')
        integrated_data = integrated_data - line_fit.eval(x=data.x)

    data.integrated_data = integrated_data

    indexs = CU.get_data_index(data.x,
                               [values.mid + params.int_entropy_range[0], values.mid + params.int_entropy_range[1]])
    if indexs[0] == indexs[1]:
        indexs = [-int(round(data.x.shape[-1] / 20)), None]
        logger.warning('Indexs averaging integrated data were the same so defaulting to last 5% of data')
    values.int_dS = np.nanmean(integrated_data[indexs[0]:indexs[1]])


def calculate_fit(data: EA_data, values: EA_value, params: EA_params, **edit_param_kwargs):
    """
    Calculates entropy fit to data.entropy_data, will use any fit parameters passed in by analysis params,
    and then will overwrite with any params passed in manually
    Args:
        data (EA_data): 1D EA data class
        values (EA_value): 1D values class
        params (EA_params): Params for analysis
        **edit_param_kwargs (dict): Kwargs for editing params before fitting. i.e.
            (param_name, value, vary, min_val, max_val)

    Returns:
        None: Just adds values.efit_info
    """
    indexs = CU.get_data_index(data.x, params.E_fit_range)
    e_fit_data = data.entropy_data[indexs[0]:indexs[1]]
    x = data.x[indexs[0]:indexs[1]]

    e_pars = E.get_param_estimates(x, e_fit_data)[0]
    if 'param_name' in params.fit_param_edit_kwargs:
        e_pars = CU.edit_params(e_pars, **params.fit_param_edit_kwargs)
    if 'param_name' in edit_param_kwargs:
        e_pars = CU.edit_params(e_pars, **edit_param_kwargs)
    efit = E.entropy_fits(x, e_fit_data, params=e_pars)[0]
    efit_info = DA.FitInfo.from_fit(efit)
    values.efit_info = efit_info
    data.data_minus_fit = data.entropy_data - efit_info.eval_fit(data.x)
    values.data_minus_fit = np.nansum(abs(data.data_minus_fit)*np.nanmean(np.diff(data.x)))


def bin_datas(data: EA_data, target_num_per_row: int):
    bin_size = np.ceil(data.x.shape[-1] / target_num_per_row)
    data.trans_data = bin_data(data.trans_data, bin_size)
    data.entropy_data = bin_data(data.entropy_data, bin_size)
    data.integrated_data = bin_data(data.integrated_data, bin_size)
    data.x = np.linspace(data.x[0], data.x[-1], int(data.x.shape[-1] / bin_size))
    return data


def _set_data(dat: DatHDF, data: EA_data):
    """Saves 1D datasets properly in HDF, more efficient for loading etc"""
    for k, v in asdict(data).items():
        if v is not None and v != []:
            dat.Other.set_data(k, v)


def get_data(dat) -> EA_data:
    """Gets 1D datasets from Dat.Other.Data, does it efficiently"""
    data = EA_data()
    for k in data.__annotations__.keys():
        if k in dat.Other.Data.keys():
            setattr(data, k, dat.Other.Data.get(k))  # Might have issues here because H5py dataset
    return data


def _set_datas(dat: DatHDF, datas: EA_datas):
    """Saves 2D datasets properly in HDF, more efficient for loading etc"""
    for name, list_datas in asdict(datas).items():
        if list_datas is not None and list_datas != []:
            for i, data in enumerate(list_datas):
                k = name + str(i)
                dat.Other.set_data(k, data)


def get_datas(dat: DatHDF) -> EA_datas:
    """Gets 2D datasets from Dat.Other.Data (i.e. if each row of data saved, not just 1D)"""
    datas = EA_datas()
    for name in datas.__annotations__.keys():
        i = 0
        list_data = getattr(datas, name)
        while True:
            k = name + str(i)
            if k in dat.Other.Data.keys():
                list_data.append(dat.Other.Data[k])
                i += 1
            else:
                break
    return datas


def save_to_dat(dat, data, values, params, uncertainties=None):
    """
    Saves each of EA_data, EA_values, EA_analysis_params to Dat.Other in a way that gets loaded automatically
    Note: data is stored as datasets, so need to call EA.get_data(dat) to get that back nicely
    Args:
        dat (DatHDF):
        data (Union[EA_data, EA_datas]): 1D EA_data class or 2D EA_datas class
        values (Union[EA_value, EA_values]): 1D EA_values class or 2D EA_values class
        params (EA_params): Params for analysis
        uncertainties (Optional[EA_value]): 1D EA_value class with uncertainties
    Returns:
        None: Saves everything in dat.Other
    """
    if isinstance(data, EA_data):
        _set_data(dat, data)
    elif isinstance(data, EA_datas):
        _set_datas(dat, data)
    else:
        logger.info(f'dat{dat.datnum} - data had class {dat.__class__} which is incorrect')

    if isinstance(values, EA_value):
        dat.Other.EA_values = values
    elif isinstance(values, EA_values):
        efits = values.efit_infos
        values.efit_infos = None
        dat.Other.EA_values = values
        for i, fit in enumerate(efits):
            name = f'efit_info_{i}'
            setattr(dat.Other, name, fit)
    else:
        logger.info(f'dat{dat.datnum} - values had class {dat.__class__} which is incorrect')

    if uncertainties is not None:
        dat.Other.EA_uncertainties = uncertainties

    dat.Other.EA_analysis_params = params
    dat.Other.time_processed = str(pd.Timestamp.now())
    dat.Other.update_HDF()


def make_datas_from_dat(dat):
    """
    Makes a datas dataclass from a single dat (i.e. data for each row)
    Args:
        dat (DatHDF): Single dat to make EA_datas from

    Returns:
        EA_datas: datas dataclass for dat
    """
    x = dat.SquareEntropy.Processed.outputs.x
    t_data = dat.SquareEntropy.Processed.outputs.cycled
    datas = make_datas(x, t_data)
    return datas


def make_datas(x, trans_data):
    """
    Makes a datas dataclass from x and trans_data only (i.e. will duplicate x to make an x for each row of trans_data,
    and will calculate entropy signal from trans_data). Note: Will not calculate integrated data because that requires
    params.
    Args:
        x (np.ndarray): 1D x_array
        trans_data (np.ndarray): Can be either 1D or 2D transition_data (already cycled... i.e. shape == ([nd], 4, len(x))

    Returns:
        EA_datas: datas class
    """
    t_data = np.array(trans_data, ndmin=3)
    e_data = np.array([entropy_signal(d) for d in t_data])
    datas = EA_datas()
    for t, e in zip(t_data, e_data):
        datas.append(EA_data(x, t, e))
    return datas


def calculate_datas(datas, params):
    """
    Calculates CT, Fit, Integrated for each data in datas using params. Returns filled values
    Args:
        datas (EA_datas):
        params (EA_params):

    Returns:
        EA_values: All fit values etc
    """
    all_values = EA_values()
    for data in datas:
        v = EA_value()
        calculate_CT_values(data, v, params)
        calculate_fit(data, v, params)
        calculate_integrated(data, v, params)
        all_values.append(v)

    return all_values


def calculate_uncertainties(datas, params):
    """
    Takes datas dataclass, fits to each individual row, and then returns the standard error (scipy.stats.sem) of the
    values stored in EA_values class. Returns in an instance of EA_value
    Args:
        datas (EA_datas): Datas to be fitted
        params (EA_params): Params to do fitting etc with

    Returns:
        EA_value: EA_value class with uncertainties instead of values
    """
    from scipy.stats import sem
    params.sub_line = False  # Doesn't really make sense with no good flat bit at the beginning for narrow data
    # (which is how I'm current calculating uncertainties)
    x = datas[0].x
    params.int_entropy_range = [x[-1] - (x[-1] - x[0]) / 20, x[-1]]  # Calc entropy value from last 5% of data

    all_values = calculate_datas(datas, params)

    uncertainties = EA_value()
    for k in set(uncertainties.__annotations__.keys()) - {'efit_info', 'fit_dS'}:  # TODO: removing fit_dS is a temp fix
        vs = getattr(all_values, k + 's', None)
        if vs is not None:
            standard_error = sem(np.array(vs), nan_policy='omit')
            setattr(uncertainties, k, standard_error)

    # TEMPORARY FIX  # TODO: make this better
    fit_dS = all_values.dSs
    if fit_dS != list():
        uncertainties.fit_dS = sem(fit_dS)

    return uncertainties


def calculate_uncertainties_from_dat(dat, params):
    """
    Wrapper for calculate_uncertainties.
    Fits to each individual row of data in dat, and then returns the standard error (scipy.stats.sem) of the
    values stored in EA_values class. Returns in an instance of EA_value
    Args:
        dat (DatHDF): Dat to get uncertainties for
        params (EA_params): Params to do fitting etc with

    Returns:
        EA_value: EA_value class with uncertainties instead of values
    """
    datas = make_datas_from_dat(dat)
    return calculate_uncertainties(datas, params)


def standard_square_process(dat: Union[DatHDF, List[DatHDF]], analysis_params: EA_params, data: EA_data = None):
    """
    Standard processing of single or two part entropy dats. Just needs EA.analysis_params to be passed in with dat(s)
    Args:
        dat (Union[Tuple[DatHDF, DatHDF], DatHDF]): Either single dat or dat pair
        analysis_params (EA_params): The analysis params to use for processing dat pair
        data (Optional[EA_data]): To be used instead of data from dats, info will be saved in dat.Other

    Returns:
        None: Info saved in dat(s).Other
    """
    if data is None:
        if isinstance(dat, Sized):
            dat: Tuple[DatHDF, DatHDF]
            if len(dat) != 2:
                raise ValueError(
                    f'dat must either be a single dat, or single pair of dats which form a two part measurement')
            for i, d in enumerate(dat):
                if hasattr(d.Logs, 'part_of'):
                    if d.Logs.part_of[0] != i + 1:
                        logger.warning(f'{len(dat)} dats passed in, the dat at position {i} reports part = '
                                       f'{d.Logs.part_of[0]}, but should be {i + 1}. Continuing anyway, '
                                       f'but things may be wrong!')
            data = data_from_dat_pair(dat, centers=None)  # Won't center automatically if LCT > -350
            datas = make_datas_from_dat(dat[1])  # Use the narrow scan dat to estimate errors
        else:
            out = dat.SquareEntropy.Processed.outputs
            data = EA_data(x=out.x, trans_data=out.averaged, entropy_data=out.entropy_signal)
            datas = make_datas_from_dat(dat)
    else:
        datas = None
        assert None not in [data.x, data.trans_data, data.entropy_data]
    values = EA_value()
    calculate_CT_values(data, values, analysis_params)  # Returns more info for debugging if necessary
    calculate_integrated(data, values, analysis_params)
    calculate_fit(data, values, analysis_params)  # Can add edit_param_kwargs here or use EA_params
    if datas is not None:
        uncertainties = calculate_uncertainties(datas, analysis_params)

    if analysis_params.bin_data is True:
        bin_datas(data, analysis_params.num_per_row)

    if not isinstance(dat, Sized):
        dats = [dat]
    else:
        dats = dat

    for d in dats:
        save_to_dat(d, data, values, analysis_params, uncertainties=uncertainties)


import plotly.graph_objects as go
import plotly.express as px


class Plots:

    @staticmethod
    def _adjust_lightness(color: str, adj: float = -0.3):
        from src.Plotting.Mpl.PlotUtil import adjust_lightness
        alpha = 1
        if color[0] == '#':
            c = color
        elif color[:4] == 'rgba':
            full_c = color[5:-1].split(',')
            c = [float(sc) / 255 for sc in full_c[0:3]]
            alpha = full_c[3]
        elif color[:3] == 'rgb':
            c = [float(sc) / 255 for sc in color[4:-1].split(',')]
        else:
            raise ValueError(f'Could not interpret {color} as a color')

        c = adjust_lightness(c, adj)
        c = [int(np.floor(sc * 255)) for sc in c]
        full_c = f'rgba({c[0]},{c[1]},{c[2]},{alpha})'
        return full_c

    @staticmethod
    def _make_hover_template(base_template='Datnum: %{customdata[0]}<br>', x_label='x=%{x:.2f}', y_label='y=%{y:.2f}',
                             additional_template=None):
        """
        Combines the base and additional_template (if passed) and returns a single template which can be used in plotly
        traces
        Args:
            base_template (str): Template string for Plotly hover data... e.g. "%{customdata[i]:.3f}<br>{...}"
            additional_template (Optional[str]): Same as above, but optional

        Returns:
            str: Combined template
        """
        hover_template = base_template + x_label + '<br>' + y_label
        if isinstance(additional_template, str):
            hover_template = hover_template + '<br>' + additional_template
        return hover_template

    @staticmethod
    def _make_customdata_getter(base_customdata: Union[List[Callable], Callable] = lambda dat: dat.datnum,
                                additional_customdata: Optional[Union[List[Callable], Callable]] = None):
        """
        Combines base an additional custom data getters, and returns a full list of customdata getters
        Args:
            base_customdata (Union[List[Callable], Callable]): Callable(s) to get data from DatHDF objects
            additional_customdata (Optional[Union[List[Callable], Callable]]): Same but optional

        Returns:
            List[Callable]: Full list of Callables which will get all custom data from DatHDF object
        """
        customdata_getters = CU.ensure_list(base_customdata)
        if callable(additional_customdata):  # If only passed in one Callable, then just turn it into a list
            additional_customdata = [additional_customdata]
        if isinstance(additional_customdata, list):
            customdata_getters.extend(additional_customdata)
        return customdata_getters

    @staticmethod
    def _get_customdata(customdata_getters: List[Callable],
                        dat: Union[List[DatHDF], DatHDF],
                        data: Optional[np.ndarray] = None):
        """
        Gets an array of customdata which matches the shape of data
        Args:
            customdata_getters (List[Callable]):  List of Callables which can get data from a DatHDF object
            dat: Either single Dat or Dats... If using single Dat, data must be provided too to get the shape 
                of customdata right
            data (Optional[np.ndarray]): This required for matching the shape of the data being plotted (customdata has 
                to have matching shape)

        Returns:
            np.ndarray: Array of customdata which matches shape of data
        """
        if not type(dat) == list:
            assert data is not None
            customdata = np.tile(np.array([f(dat) for f in customdata_getters]),
                                 (data.shape[-1], 1))  # Same outer dim as data 
        else:
            customdata = np.array([[f(d) for f in customdata_getters] for d in dat])
        return customdata

    @staticmethod
    def sorted(dats: List[DatHDF], which_sort, which_x, which_y, sort_array=None, sort_tol=None, mode='markers',
               uncertainties='fill', fig=None, get_additional_data=None, additional_hover_template=None):
        """
        Returns a plotly figure with multiple traces. Which_sort determines how data is grouped together.
        Args:
            dats (List[DatHDF]): Dats can be multi part (This will ignore anything but part_1)
            which_sort (str): What to sort by
            which_x (str): What to use on x_axis
            which_y (str): What to use on y_axis
            sort_array (Optional[Union[list, tuple, np.ndarray]]): List of values to sort by
            sort_tol (Optional[float]):  How close the value from the dat has to be to the sort_array values
            mode (str): Whether to use markers and/or lines (e.g. 'markers+lines')
            uncertainties (Optional[str]): Whether to add uncertainties to plot. Can be (None, 'fill', 'bars')
            fig (go.Figure): Modify the figure passed in

        Returns:
            go.Figure: Plotly figure with traces
        """

        SORT_KEYS = ('lct', 'temp', 'field', 'hqpc', 'rcb')
        X_KEYS = list(SORT_KEYS) + ['time']
        Y_KEYS = ('fit_ds', 'int_ds', 'dt', 'data_minus_fit', 'dmf', 'data_minus_fit_scaled', 'dmfs')

        colors = px.colors.DEFAULT_PLOTLY_COLORS  # Have to specify so I can add error fill with the same background color

        which_sort, which_x, which_y = which_sort.lower(), which_x.lower(), which_y.lower()

        if which_sort not in SORT_KEYS:
            raise ValueError(f'Which_sort must be one of: {SORT_KEYS}')
        if which_x not in X_KEYS:
            raise ValueError(f'Which_x must be one of: {X_KEYS}')
        if which_y not in Y_KEYS:
            raise ValueError(f'Which_y must be one of: {Y_KEYS}')

        if which_sort == 'temp':
            name = 'Temp'
            units = 'mK'
            get_val = lambda dat: dat.Logs.temps.mc * 1000
            tol = 10
        elif which_sort == 'field':
            name = 'Field'
            units = 'mT'
            # array = np.linspace(-21, -19, 21)
            get_val = lambda dat: dat.Logs.magy.field
            tol = 0.2
        elif which_sort == 'lct':
            name = 'LCT'
            units = 'mV'
            # array = np.linspace(-460, -380, 5)
            get_val = lambda dat: dat.Logs.fds['LCT']
            tol = 5
        elif which_sort == 'hqpc':
            name = 'HQPC bias'
            units = 'mV'
            # array = np.linspace(117.5, 122.5, 11)
            get_val = lambda dat: dat.AWG.AWs[0][0][1]
            tol = 0.2
        elif which_sort == 'rcb':
            name = 'RCB'
            units = 'mV'
            get_val = lambda dat: dat.Logs.bds['RCB']
        else:
            raise ValueError

        if sort_tol is not None:
            tol = sort_tol
        if sort_array is not None:
            array = sort_array
        else:
            array = set(CU.my_round(np.array([get_val(dat) for dat in dats]), prec=3, base=tol/2))

        if which_x == 'lct':
            get_x = lambda dat: dat.Logs.fds['LCT']
            x_title = 'LCT /mV'
        elif which_x == 'field':
            get_x = lambda dat: dat.Logs.magy.field
            x_title = 'Field /mT'
        elif which_x == 'temp':
            get_x = lambda dat: dat.Logs.temps.mc * 1000
            x_title = 'MC Temp /mK'
        elif which_x == 'hqpc':
            get_x = lambda dat: dat.AWG.AWs[0][0][1]
            x_title = 'HQPC bias /mV'
        elif which_x == 'time':
            get_x = lambda dat: pd.Timestamp(dat.Logs.time_completed)
            x_title = 'Time'
        elif which_x == 'rcb':
            get_x = lambda dat: dat.Logs.bds['RCB']
            x_title = 'RCB /mV'
        else:
            raise ValueError

        if which_y == 'dt':
            get_y = lambda values: values.dTs
            y_title = 'dT /mV'
            get_u = get_y
        elif which_y == 'amp':
            get_y = lambda values: values.amps
            y_title = 'Amplitude /nA'
            get_u = get_y
        elif which_y == 'fit_ds':
            get_y = lambda values: values.dSs
            y_title = 'Fit Entropy /kB'
            get_u = lambda us: us.fit_dSs
        elif which_y == 'int_ds':
            get_y = lambda values: values.int_dSs
            y_title = 'Integrated Entropy /kB'
            get_u = get_y
        elif which_y in ('data_minus_fit', 'dmf'):
            get_y = lambda values: values.data_minus_fits
            y_title = 'Data Minus Fit /(nA*mV)'
            get_u = get_y
        elif which_y in ('data_minus_fit_scaled', 'dmfs'):
            get_y = lambda values: np.array(values.data_minus_fits)/np.array(values.dTs)
            y_title = 'Data Minus Fit/dT /arb'
            get_u = get_y
        else:
            raise ValueError

        dats = [dat for dat in dats if dat.Logs.part_of is None or dat.Logs.part_of[0] == 1]

        # General which relies on specific
        custom_data_getters = Plots._make_customdata_getter(base_customdata=lambda dat: dat.datnum,
                                                            additional_customdata=get_additional_data)
        hover_template = Plots._make_hover_template(base_template='Datnum: %{customdata[0]}<br>',
                                                    x_label=x_title + ': %{x:.3f}',
                                                    y_label=y_title + ': %{y:.3f}',
                                                    additional_template=additional_hover_template)

        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: Repeats along transition sorted by {name}',
                xaxis_title=x_title, yaxis_title=y_title)

        for i, val in enumerate(array):
            ds = [dat for dat in dats if np.isclose(get_val(dat), val, atol=tol)]
            values = EA_values.from_dats(ds)
            x = np.array([get_x(dat) for dat in ds])
            y = np.array(get_y(values))
            if None in y:
                continue
            customdata = Plots._get_customdata(custom_data_getters, ds)
            color = colors[i % len(colors)]
            if uncertainties is not None:
                u_values = EA_values.from_dats(ds, uncertainty=True)
                us = np.array(get_u(u_values))
                error = dict(error_y=dict(
                    type='data',  # value of error bar given in data coordinates
                    array=us,
                    visible=True))
            else:
                error = dict()
            trace = go.Scatter(x=x, y=y, name=f'{val:.2f}{units}', mode=mode, customdata=customdata,
                               hovertemplate=hover_template, line=dict(color=color), legendgroup=f'{val:.2f}{units}',
                               **error)
            fig.add_trace(trace)
            if uncertainties is not None:
                if None in us:
                    continue
                if uncertainties == 'fill':
                    color = f'rgba{color[3:-1]}, 0.3)'  # Adding alpha to color
                    y_upper = y + us
                    y_lower = y - us
                    fig.add_trace(go.Scatter(
                        x=list(x) + list(x)[::-1],  # x, then x reversed
                        y=list(y_upper) + list(y_lower)[::-1],  # upper, then lower reversed
                        fill='toself',
                        fillcolor=color,
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=f'{val:.2f}{units}'
                    ))
        return fig

    @staticmethod
    def waterfall(dats, which='transition', mode='lines', add_fits=False, shift_per=0.025,
                  fig=None, get_additional_data=None, additional_hover_template=None):
        """
        Makes waterfall plot (or on top of each other) from dats. Returns a plotly go.Figure, so could be used as
        a starting point and be modified. Or can pass in a figure to have this add data but not affect labels etc
        Args:
            dats (List[DatHDF]):
            which (str):
            mode (str):
            add_fits (bool):
            shift_per (float):
            fig (go.Figure):
            get_additional_data (Optional[List[func]]): List of Callables which take dat as an argument and return
                    desired data to be shown in hover data
            additional_hover_template (Optional[str]): Template string to show the custom data...
                    Use %{customdata[i]:.1f} to show where 'i' starts at 1 <<<<!!! NOT 0 (because 0 is used for datnum)

        Returns:
            go.Figure: Plotly figure with traces added. Get to traces using fig.data[]
        """

        PLOT_TYPES = ('transition', 'entropy', 'integrated', 'data_minus_fit', 'dmf')
        which = which.lower()
        dats = [dat for dat in dats if dat.Logs.part_of is None or dat.Logs.part_of[0] == 1]
        datas = [get_data(dat) for dat in dats]
        valuess = [getattr(dat.Other, 'EA_values') for dat in dats]

        # General/Defaults
        get_x = lambda data: data.x
        x_label = 'LP*200 /mV'
        y_label = 'Entropy /kB'
        get_fits = 'NOT SET'  # Replace with lambda ... to use below
        label_prefix = lambda dat: f'Dat{dat.datnum}'
        colors = px.colors.DEFAULT_PLOTLY_COLORS  # Have to specify so I can add fit with same color
        get_scaling = lambda dat: 1

        # Transition Plot Params
        if which == 'transition':
            name = 'Transition'
            get_y = lambda data: data.trans_data[:]
            get_x = lambda data: np.array([data.x] * 4)
            label = ('v0_0', 'vp', 'v0_1', 'vm')
            y_label = 'Current /nA'

        # Entropy plot Params
        elif which == 'entropy':
            name = 'Fit Entropy'
            get_y = lambda data: data.entropy_data[:]
            get_fits = lambda data, values: values.efit_info.eval_fit(x=data.x)
            label = 'Entropy signal'

        # Integrated Plot Params
        elif which == 'integrated':
            name = 'Integrated Entropy'
            get_y = lambda data: data.integrated_data[:]
            label = 'Integrated Signal'

        # Data - Fit (Bumps)
        elif which in ('data_minus_fit', 'dmf'):
            name = 'Data subtract Fit'
            get_y = lambda data: data.data_minus_fit[:]
            label = 'Data minus fit'
            y_label = 'Delta E /(nA*mV)'

        # Data - Fit scaled by dT
        elif which in ('data_minus_fit_scaled', 'dmfs'):
            name = 'Data subtract Fit over dT'
            get_y = lambda data: data.data_minus_fit[:]
            get_scaling = lambda dat: 1/dat.Other.EA_values.dT
            label = 'Data minus fit/dT'
            y_label = 'Delta E /(nA*mV)'
        else:
            raise ValueError(f"Got {which} for 'which', but 'which' must be in {PLOT_TYPES}")

        # General which relies on specific
        custom_data_getters = Plots._make_customdata_getter(base_customdata=lambda dat: dat.datnum,
                                                            additional_customdata=get_additional_data)
        hover_template = Plots._make_hover_template(base_template='Datnum: %{customdata[0]}<br>',
                                                    x_label=x_label + ': %{x:.3f}',
                                                    y_label=y_label + ': %{y:.3f}',
                                                    additional_template=additional_hover_template)

        if fig is None:
            fig = go.Figure()
            fig.update_layout(xaxis_title=x_label,
                              yaxis_title=y_label,
                              title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: {name}')
        for i, (dat, data, values) in enumerate(zip(dats, datas, valuess)):

            shift_y = i * shift_per
            xs = np.atleast_2d(get_x(data))  # In case multiple lines per plot, make it always work as if it's 2D
            scaling = get_scaling(dat)
            ys = np.atleast_2d(get_y(data))*scaling  # In case multiple lines per plot, make it always work as if it's 2D
            labels = CU.ensure_list(label)
            lp = label_prefix(dat)
            customdata = Plots._get_customdata(custom_data_getters, dat, xs[0])
            for j, (x, y, l) in enumerate(zip(xs, ys, labels)):
                color = colors[(i + j) % len(colors)]
                fig.add_trace(go.Scatter(x=x, y=y + shift_y, customdata=customdata, hovertemplate=hover_template,
                                         name=f'{lp}_{l}', mode=mode, legendgroup=f'{lp}_{l}',
                                         line=dict(color=color)))
            if add_fits is True:
                fits = np.atleast_2d(get_fits(data, values))
                for j, (x, fit, l) in enumerate(zip(xs, fits, labels)):
                    color = colors[(i + j) % len(colors)]
                    color = Plots._adjust_lightness(color)
                    fig.add_trace(go.Scatter(x=data.x, y=fit + shift_y, name=f'{lp}_{l} Fit', legendgroup=lp, mode=mode,
                                             line=dict(color=color)))

        return fig


if __name__ == '__main__':
    from src.DatObject.Make_Dat import DatHandler as DH

    # dats = DH.get_dats((3456, 3457 + 1), overwrite=False)
    # dats = DH.get_dats((4267, 4268 + 1))  # Test  0->1 transition
    # dats.extend(DH.get_dats((4275, 4276 + 1)))  # Test  Accidentally 1->2 transition?!!
    # dats.extend(DH.get_dats((4278, 4279 + 1)))  # Test  Accidentally 1->2 transition?!!
    # dats.extend(DH.get_dats((4280, 4281 + 1)))
    dats = DH.get_dats((4855, 4910 + 1))

    # analysis_params = EA_params(bin_data=True, num_per_row=500,
    #                             sub_line=True, sub_line_range=(-4000, -600),
    #                             int_entropy_range=(300, 400),
    #                             allowed_amp_range=(0.6, 1.500001), default_amp=1.022,
    #                             allowed_dT_range=(0.001, 35.1), default_dT=3.87,
    #                             CT_fit_range=(-250, 250),
    #                             fit_param_edit_kwargs=dict(),
    #                             E_fit_range=(-100, 200))
    #
    # standard_square_process(dats, analysis_params)
    #
    # x = dats[1].SquareEntropy.Processed.outputs.x
    # trans_data_2d = dats[1].SquareEntropy.Processed.outputs.cycled
    # values = EA_value()
    # us = calculate_uncertainties_from_dat(dats[1], params)
    # print(EA_values.from_dats(dats, uncertainty=True))
    # Plots.waterfall(dats, which='transition', mode='lines', add_fits=False, shift_per=0.025)
    Plots.sorted(dats, which_sort='temp', which_x='time', which_y='dt')