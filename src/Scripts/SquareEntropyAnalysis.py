from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, TYPE_CHECKING, Tuple
import numpy as np
from src.CoreUtil import bin_data
import src.CoreUtil as CU
from src.DatObject.Attributes.SquareEntropy import average_2D, entropy_signal, scaling, IntegratedInfo, integrate_entropy
from src.DatObject.Attributes.Transition import transition_fits, i_sense
from src.DatObject.Attributes import Entropy as E, DatAttribute as DA
import re
import lmfit as lm
import logging
import pandas as pd

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF


def merge_dat_parts(dats, centers=None):  # TODO: This could be a lot more general and not only work for SE
    """
    Merges two part Square entropy dats.
    Args:
        dats (List[DatHDF]):
        centers (Optional[np.ndarray]):

    Returns:
        Tuple[np.ndarray, np.ndarray]: full_x, full_cycled_data
    """
    assert len(dats) == 2
    p1_dat, p2_dat = None, None
    for dat in dats:
        comments = dat.Logs.comments.split(',')
        comments = [com.strip() for com in comments]
        part_comment = [com for com in comments if re.match('part*', com)][0]
        part_num = int(re.search('(?<=part)\d+', part_comment).group(0))
        dat.Other.part_num = part_num
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

    p2_data = np.pad(p2_data, ((0, 0), (0, 0), (idx, full_x.shape[0] - p2_x.shape[0] - idx)), mode='constant',
                     constant_values=np.nan)

    full_data = np.concatenate((p1_data, p2_data), axis=0)
    full_x, full_avg_data = average_2D(full_x, full_data, avg_nans=True, centers=centers)
    return full_x, full_avg_data


@dataclass
class EA_params:
    bin_data: bool
    num_per_row: int

    sub_line: bool
    sub_line_range: Tuple[float, float]

    int_entropy_range: Tuple[float, float]

    allowed_amp_range: Tuple[float, float]
    default_amp: float
    allowed_dT_range: Tuple[float, float]
    default_dT: float

    CT_fit_range: Tuple[Optional[float], Optional[float]] = field(default=(None, None))  # Useful if fitting with two part dat

    fit_param_edit_kwargs: dict = field(default_factory=dict)

@dataclass
class EA_datas:
    xs: list = field(default_factory=list)
    trans_datas: list = field(default_factory=list)
    entropy_datas: list = field(default_factory=list)
    integrated_datas: list = field(default_factory=list)

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
        return datas

    def add_fit_to_entropys(self, fits: List[DA.FitInfo]):
        self.entropy_datas = [[fit.eval_fit(x), edata] for fit, x, edata in zip(fits, self.xs, self.entropy_datas)]

@dataclass
class EA_data:
    x: np.ndarray = field(default=None)
    trans_data: np.ndarray = field(default=None)
    entropy_data: np.ndarray = field(default=None)
    integrated_data: np.ndarray = field(default=None)

@dataclass
class EA_titles:
    ids: list = field(default_factory=list)
    trans: list = field(default_factory=list)
    entropy: list = field(default_factory=list)
    integrated: list = field(default_factory=list)


def data_from_dat_pair(dat_pair: List[DatHDF, DatHDF], centers=None) -> EA_data:
    """Takes dat pair and combines them into 1D averaged dataset"""
    dat = dat_pair[0]
    if centers is None:
        if dat.Logs.fds["LCT"] > -350:
            centers = [0] * len(dat.Data.y_array)

    x, averaged_data = merge_dat_parts(dat_pair, centers=centers)  # centers = None finds center with transition fit
    entropy_data = entropy_signal(averaged_data)
    data = EA_data(x=x, trans_data=averaged_data, entropy_data=entropy_data)
    return data


@dataclass
class DataCTvalues:
    tcs: List[float] = field(default_factory=list)
    ths: List[float] = field(default_factory=list)
    amps: List[float] = field(default_factory=list)
    mids: List[float] = field(default_factory=list)


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
    efit_info: DA.FitInfo = field(default=None)


def calculate_CT_values(data: EA_data, values: EA_value, params: EA_params) -> DataCTvalues:
    """
    Calculates the charge transition fit stuff, stores the interesting final values in 'values', but
    also returns ct_values which has each individual fit value used
    Args:
        data (EA_data): Entropy Analysis dataclass
        values (EA_value): Entropy Analysis values
        params (EA_params): Entropy Analysis params

    Returns:
        (DataCTvalues): Each fit value used to fill 'values', mostly for debugging
    """
    t_fit_data = data.trans_data[params.CT_fit_range[0]:params.CT_fit_range[1]]
    x = data.x[params.CT_fit_range[0]:params.CT_fit_range[1]]
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

    mid = np.average(ct_values.mids)
    amp = np.average(ct_values.amps)
    dT = values.th-values.tc

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
    values.dx = float(np.mean(np.diff(data.x)))
    values.sf = scaling(dt=values.dT, amplitude=values.amp, dx=values.dx)
    integrated_data = integrate_entropy(data.entropy_data, values.sf)

    if params.sub_line:
        line = lm.models.LinearModel()
        indexs = CU.get_data_index(data.x, [values.mid + params.sub_line_range[0], values.mid + params.sub_line_range[1]])
        line_fit = line.fit(integrated_data[indexs[0]:indexs[1]], x=data.x[indexs[0]:indexs[1]], nan_policy='omit')
        integrated_data = integrated_data - line_fit.eval(x=data.x)

    data.integrated_data = integrated_data

    indexs = CU.get_data_index(data.x, [values.mid + params.int_entropy_range[0], values.mid + params.int_entropy_range[1]])
    values.int_dS = np.mean(integrated_data[indexs[0]:indexs[1]])


def calculate_fit(data: EA_data, values: EA_value, params: EA_params, **edit_param_kwargs):
    e_pars = E.get_param_estimates(data.x, data.entropy_data)[0]
    if 'param_name' in params.fit_param_edit_kwargs:
        e_pars = CU.edit_params(e_pars, **params.fit_param_edit_kwargs)
    if 'param_name' in edit_param_kwargs:
        e_pars = CU.edit_params(e_pars, **edit_param_kwargs)
    efit = E.entropy_fits(data.x, data.entropy_data, params=e_pars)[0]
    efit_info = DA.FitInfo.from_fit(efit)
    values.efit_info = efit_info


def bin_datas(data: EA_data, target_num_per_row: int):
    bin_size = np.ceil(data.x.shape[-1] / target_num_per_row)
    data.trans_data = bin_data(data.trans_data, bin_size)
    data.entropy_data = bin_data(data.entropy_data, bin_size)
    data.integrated_data = bin_data(data.integrated_data, bin_size)
    data.x = np.linspace(data.x[0], data.x[-1], int(data.x.shape[-1] / bin_size))
    return data


def set_data(dat: DatHDF, data: EA_data):
    """Saves datasets properly in HDF, more efficient for loading etc"""
    for k, v in asdict(data).items():
        if v is not None:
            dat.Other.set_data(k, v)


def get_data(dat) -> EA_data:
    """Gets datasets from Dat.Other.Data, does it efficiently"""
    data = EA_data()
    for k in data.__annotations__.keys():
        if k in dat.Other.Data.keys():
            setattr(data, k, dat.Other.Data[k])  # Might have issues here because H5py dataset
    return data


def standard_square_process(dat: Union[DatHDF, List[DatHDF]], analysis_params: EA_params, data: EA_data = None):
    if data is None:
        if type(dat) == list:
            if len(dat) != 2:
                raise ValueError(f'dat must either be a single dat, or single pair of dats which form a two part measurement')
            data = data_from_dat_pair(dat, centers=None)  # Won't center automatically if LCT > -350
        else:
            out = dat.SquareEntropy.Processed.outputs
            data = EA_data(x=out.x, trans_data=out.averaged, entropy_data=out.entropy_signal)
    else:
        assert None not in [data.x, data.trans_data, data.entropy_data]
    values = EA_value()
    calculate_CT_values(data, values, analysis_params)  # Returns more info for debugging if necessary
    calculate_integrated(data, values, analysis_params)
    calculate_fit(data, values, analysis_params)  # Can add edit_param_kwargs here or use EA_params

    if analysis_params.bin_data is True:
        bin_datas(data, analysis_params.num_per_row)

    if type(dat) != list:
        dats = [dat]
    else:
        dats = dat

    for dat in dats:
        set_data(dat, data)
        dat.Other.EA_values = values
        dat.Other.EA_analysis_params = analysis_params
        dat.Other.time_processed = str(pd.Timestamp.now())
        dat.Other.update_HDF()


