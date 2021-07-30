from __future__ import annotations
import inspect
from dataclasses import dataclass, InitVar, field
from hashlib import md5
from typing import Union, Optional, Callable, Any, TYPE_CHECKING, Tuple

import re
import h5py
import lmfit as lm
import numpy as np
import pandas as pd
import logging

import scipy
from scipy.interpolate import RectBivariateSpline

from src import core_util as CU, useful_functions as U
from src.dat_object.Attributes.SquareEntropy import Output as SeOutput, square_wave_time_array, centers_from_fits
from src.dat_object.Attributes.Transition import get_param_estimates, get_transition_function
from src.hdf_util import params_from_HDF, params_to_HDF, NotFoundInHdfError, DatDataclassTemplate

if TYPE_CHECKING:
    from src.dat_object.dat_hdf import DatHDF

logger = logging.getLogger(__name__)


class Values(object):
    """Object to store Init/Best values in and stores Keys of those values in self.keys"""

    def __init__(self):
        self.keys = []

    def __getattr__(self, item):
        if item.startswith('__') or item.startswith(
                '_') or item == 'keys':  # So don't complain about things like __len__
            return super().__getattribute__(item)  # Come's here looking for Ipython variables
        else:
            if item in self.keys:
                return super().__getattribute__(item)
            else:
                msg = f'{item} does not exist. Valid keys are {self.keys}'
                print(msg)
                logger.warning(msg)
                return None

    def get(self, item, default=None):
        if item in self.keys:
            val = self.__getattr__(item)
        else:
            val = default
        if default is not None and val is None:
            return default
        return val

    def __setattr__(self, key, value):
        if key.startswith('__') or key.startswith('_') or key == 'keys' or not isinstance(value, (
                np.number, float, int, type(None))):  # So don't complain about
            # things like __len__ and don't keep key of random things attached to class
            super().__setattr__(key, value)
        else:  # probably is something I want the key of
            self.keys.append(key)
            super().__setattr__(key, value)

    def __repr__(self):
        string = ''
        for key in self.keys:
            v = getattr(self, key)
            if v is not None:
                string += f'{key}={self.__getattr__(key):.5g}\n'
            else:
                string += f'{key}=None\n'
        return string

    def to_df(self):
        df = pd.DataFrame(data=[[self.get(k) for k in self.keys]], columns=[k for k in self.keys])
        return df


@dataclass
class FitInfo(DatDataclassTemplate):
    params: Union[lm.Parameters, None] = None
    init_params: lm.Parameters = None
    func_name: Union[str, None] = None
    func_code: Union[str, None] = None
    fit_report: Union[str, None] = None
    model: Union[lm.Model, None] = None
    best_values: Union[Values, None] = None
    init_values: Union[Values, None] = None
    success: bool = None
    hash: Optional[int] = None

    # Will only exist when set from fit, or after recalculate_fit
    fit_result: Union[lm.model.ModelResult, None] = None

    @property
    def reduced_chi_sq(self):
        return float(re.search(r'(?:reduced chi-square\s*=\s)(.*)', self.fit_report).groups()[0])

    def init_from_fit(self, fit: lm.model.ModelResult, hash_: Optional[int] = None):
        """Init values from fit result"""
        if fit is None:
            logger.warning(f'Got None for fit to initialize from. Not doing anything.')
            return None
        assert isinstance(fit, lm.model.ModelResult)
        self.params = fit.params
        self.init_params = fit.init_params
        self.func_name = fit.model.func.__name__

        #  Can't get source code when running from deepcopy (and maybe other things will break this)
        try:
            func_code = inspect.getsource(fit.model.func)
        except OSError:
            if self.func_code is not None:
                func_code = '[WARNING]: might not be correct as fit was re run and could not get source code: ' \
                            '' + self.func_code
            else:
                logger.warning('Failed to get source func_code and no existing func_code')
                func_code = 'Failed to get source code due to OSError'
        self.func_code = func_code

        self.fit_report = fit.fit_report()
        self.success = fit.success
        self.model = fit.model
        self.best_values = Values()
        self.init_values = Values()
        for key in self.params.keys():
            par = self.params[key]
            self.best_values.__setattr__(par.name, par.value)
            self.init_values.__setattr__(par.name, par.init_value)
        self.hash = hash_
        self.fit_result = fit

    def init_from_hdf(self, group: h5py.Group):
        """Init values from HDF file"""
        self.params = params_from_HDF(group)
        self.init_params = params_from_HDF(group.get('init_params'), initial=True)
        self.func_name = group.attrs.get('func_name', None)
        self.func_code = group.attrs.get('func_code', None)
        self.fit_report = group.attrs.get('fit_report', None)
        self.model = lm.models.Model(self._get_func())
        self.success = group.attrs.get('success', None)

        self.best_values = Values()
        self.init_values = Values()
        for key in self.params.keys():
            par = self.params[key]
            self.best_values.__setattr__(par.name, par.value)
            self.init_values.__setattr__(par.name, par.init_value)

        temp_hash = group.attrs.get('hash')
        if temp_hash is not None:
            self.hash = int(temp_hash)
        else:
            self.hash = None
        self.fit_result = None

    def save_to_hdf(self, parent_group: h5py.Group, name: Optional[str] = None):
        if name is None:
            name = self._default_name()
        parent_group = parent_group.require_group(name)

        if self.params is None:
            logger.warning(f'No params to save for {self.func_name} fit. Not doing anything')
            return None
        params_to_HDF(self.params, parent_group)
        params_to_HDF(self.init_params, parent_group.require_group('init_params'))
        parent_group.attrs['description'] = 'FitInfo'  # Overwrites what params_to_HDF sets
        parent_group.attrs['func_name'] = self.func_name
        parent_group.attrs['func_code'] = self.func_code
        parent_group.attrs['fit_report'] = self.fit_report
        parent_group.attrs['success'] = self.success
        if self.hash is not None:
            parent_group.attrs['hash'] = int(self.hash)

    def _get_func(self):
        """Did not initially enforce having a good way to get back the lm.model when loading from hdf, so this is
        the workaround... Let FitInfo know what the functions are for the saved function name.
        Note: name part must match function name exactly"""
        # return HDU.get_func(self.func_name, self.func_code)
        from src.dat_object.Attributes.Transition import i_sense, i_sense_strong, i_sense_digamma, i_sense_digamma_quad, \
            i_sense_digamma_amplin
        from src.analysis_tools.nrg import NRG_func_generator
        from src.dat_object.Attributes.Entropy import entropy_nik_shape
        funcs = {
            'i_sense': i_sense,
            'i_sense_strong': i_sense_strong,
            'i_sense_digamma': i_sense_digamma,
            'i_sense_digamma_quad': i_sense_digamma_quad,
            'i_sense_digamma_amplin': i_sense_digamma_amplin,
            'entropy_nik_shape': entropy_nik_shape,
            'nrg_func': NRG_func_generator(which='i_sense')
        }
        if self.func_name in funcs:
            return funcs[self.func_name]
        else:
            raise KeyError(f'{self.func_name} not a recongnized function in '
                           f'src.DatObject.Attributes.DatAttribute.FitInfo')

    def eval_fit(self, x: np.ndarray):
        """Return best fit for x array using params"""
        return self.model.eval(self.params, x=x)

    def eval_init(self, x: np.ndarray):
        """Return init fit for x array using params"""
        init_pars = CU.edit_params(self.params, list(self.params.keys()),
                                   [par.init_value for par in self.params.values()])
        return self.model.eval(init_pars, x=x)

    def recalculate_fit(self, x: np.ndarray, data: np.ndarray, auto_bin=False, min_bins=1000):
        """Fit to data with x array and update self"""
        assert data.ndim == 1
        data, x = CU.remove_nans(data, x)
        if auto_bin is True and len(data) > min_bins:
            logger.info(f'Binning data of len {len(data)} into {min_bins} before fitting')
            x, data = CU.bin_data([x, data], round(len(data) / min_bins))
        fit = self.model.fit(data.astype(np.float32), self.params, x=x, nan_policy='omit')
        self.init_from_fit(fit, self.hash)

    def edit_params(self, param_names=None, values=None, varys=None, mins=None, maxs=None):
        self.params = CU.edit_params(self.params, param_names, values, varys, mins, maxs)

    def to_df(self):
        val_df = self.best_values.to_df()
        val_df['success'] = self.success
        return val_df

    def __hash__(self):
        if self.hash is None:
            raise AttributeError(f'hash value stored as None so hashing not supported')
        return int(self.hash)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return hash(other) == hash(self)
        return False

    def __repr__(self):
        return self.fit_report

    def __getstate__(self):
        """For dumping to pickle"""
        # logger.warning(f'FitInfo object is not picklable because of many things which are stored... Override this and __setstate__ in order to make it picklable')
        return self.__dict__
        # return super().__getstate__()
        # raise NotImplementedError(f'FitInfo object is not picklable because of many things which are stored... Override this and __setstate__ in order to make it picklable')

    def __setstate__(self, state):
        """For loading from pickle"""
        self.__dict__.update(state)


    @classmethod
    def from_fit(cls, fit, hash_: Optional[int] = None):
        """Use FitIdentifier to generate hash (Should be done before binning data to be able to check if
        matches before doing expensive processing)"""
        inst = cls()
        inst.init_from_fit(fit, hash_)
        return inst

    @classmethod
    def from_hdf(cls, parent_group: h5py.Group, name: str = None):
        if name is None:
            name = cls._default_name()
        fg = parent_group.get(name)
        if fg is None:
            raise NotFoundInHdfError(f'{name} not found in {parent_group.name}')
        inst = cls()
        inst.init_from_hdf(fg)
        return inst


@dataclass
class FitIdentifier:
    initial_params: lm.Parameters
    func: Callable  # Or should I just use func name here? Or func code?
    data: InitVar[np.ndarray]
    data_hash: str = field(init=False)

    def __post_init__(self, data: np.ndarray):
        assert isinstance(self.initial_params, lm.Parameters)
        self.data_hash = self._hash_data(data)

    @staticmethod
    def _hash_data(data: np.ndarray):
        if data.ndim == 1:
            data = data[~np.isnan(data)]  # Because fits omit NaN data so this will make data match the fit data.
        return md5(data.tobytes()).hexdigest()

    def __hash__(self):
        """The default hash of FitIdentifier which will allow comparison between instances
        Using hashlib hashes makes this deterministic rather than runtime specific, so can compare to saved values
        """
        pars_hash = self._hash_params()
        func_hash = self._hash_func()
        data_hash = self.data_hash
        h = md5(pars_hash.encode())
        h.update(func_hash.encode())
        h.update(data_hash.encode())
        return int(h.hexdigest(), 16)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if hash(self) == hash(other):
                return True
        return False

    def _hash_params(self) -> str:
        h = md5(str(sorted(self.initial_params.valuesdict().items())).encode())
        return h.hexdigest()

    def _hash_func(self) -> str:
        # hash(self.func)   # Works pretty well, but if function is executed later even if it does the same thing it
        # will change
        h = md5(str(self.func.__name__).encode())
        return h.hexdigest()

    def generate_name(self):
        """ Will be some thing reproducible and easy to read. Note: not totally guaranteed to be unique."""
        return str(hash(self))[0:5]


def calculate_fit(x: np.ndarray, data: np.ndarray, params: lm.Parameters, func: Callable[[Any], float],
                  auto_bin=True, min_bins=1000, generate_hash=False,
                  warning_id: Optional[str] = None,
                  method: str = 'leastsq',
                  ) -> FitInfo:
    """
    Calculates fit on data (Note: assumes that 'x' is the independent variable in fit_func)
    Args:
        x (np.ndarray): x_array (Note: fit_func should have variable with name 'x')
        data (np.ndarray): Data to fit
        params (lm.Parameters): Initial parameters for fit
        func (Callable): Function to fit to
        auto_bin (bool): if True will bin data into >= min_bins
        min_bins: How many bins to use for binning (actual num will lie between min_bins >= actual > min_bins*1.5)
        generate_hash: Whether to hash the data and fit params for comparison in future
        warning_id: String to use warning messages if fits aren't completely successful

    Returns:
        (FitInfo): FitInfo instance (with FitInfo.fit_result filled)
    """
    def sanitize_params(pars: lm.Parameters) -> lm.Parameters:
        # for par in pars:
        #     pars[par].value = np.float32(pars[par].value)  # VERY infrequently causes issues for calculating
        #     # uncertainties with np.float64 dtype
        return pars

    model = lm.model.Model(func)
    if generate_hash:
        hash_ = hash(FitIdentifier(params, func, data))  # Needs to be done BEFORE binning data.
    else:
        hash_ = None

    if auto_bin and data.shape[-1] > min_bins*2:  # between 1-2x min_bins won't actually end up binning
        bin_size = int(np.floor(data.shape[-1] / min_bins))  # Will end up with >= self.AUTO_BIN_SIZE pts
        x, data = [CU.bin_data_new(arr, bin_x=bin_size) for arr in [x, data]]

    params = sanitize_params(params)
    try:
        fit = FitInfo.from_fit(
            model.fit(data.astype(np.float32), params, x=x.astype(np.float32), nan_policy='omit', method=method), hash_)
        if fit.fit_result.covar is None and fit.success is True:  # Failed to calculate uncertainties even though fit
            # was successful
            logger.warning(f'{warning_id}: Uncertainties failed')
        elif fit.success is False:
            logger.warning(f'{warning_id}: A fit failed')
    except TypeError as e:
        logger.error(f'{e} while fitting {warning_id}')
        fit = None
    return fit


@dataclass
class CalculatedFit:
    x: np.ndarray
    data: np.ndarray
    fit: FitInfo


@dataclass
class CalculatedTransitionFit(CalculatedFit):
    pass


@dataclass
class CalculatedEntropyFit(CalculatedFit):
    output: SeOutput
    pass


def calculate_se_output(dat: DatHDF, rows, csq_mapped,
                        center_func_name: str,
                        setpoint_start: Optional[float]) -> SeOutput:
    """
    Calculate SE output using EXISTING center fits... I.e. no fitting run in this, just the SE processing and averaging
    using existing center fits.

    Args:
        dat (): SE entropy dat
        rows (): Rows to process between (Nones are beginning or end)
        csq_mapped (): Whether to use CSQ mapped data
        center_func_name (): Name of func used for centering (these fits will be loaded from)
        setpoint_start (): Amount of time after each setpoint to throw out

    Returns:
        SeOutput with all relevant data filled
    """

    def get_setpoint_ids(d: DatHDF, start_time, fin_time=None):
        sps = [start_time, fin_time]
        sp_times = square_wave_time_array(d.SquareEntropy.square_awg)
        start, fin = [U.get_data_index(sp_times, sp) for sp in sps]
        return start, fin

    def get_data(d: DatHDF, rs, csq):
        s, f = rs
        if csq:
            data_ = d.Data.get_data('csq_mapped')[s:f]
        else:
            data_ = d.Transition.get_data('i_sense')[s:f]
        return data_

    centers = get_centers(dat, center_func_name=center_func_name, rows=rows, se_data=True)

    sp_start, sp_fin = get_setpoint_ids(dat, setpoint_start, None)
    x = dat.Data.get_data('x')
    data = get_data(dat, rows, csq_mapped)
    inputs = dat.SquareEntropy.get_Inputs(name=None, x_array=x, i_sense=data, centers=centers,
                                          save_name=None)

    process_params = dat.SquareEntropy.get_ProcessParams(name=None,
                                                         setpoint_start=sp_start, setpoint_fin=sp_fin,
                                                         transition_fit_func=None,  # Don't need to center again
                                                         transition_fit_params=None,  # Don't need to center again
                                                         save_name=None,  # Do not save
                                                         )
    out = dat.SquareEntropy.get_Outputs(inputs=inputs, process_params=process_params,
                                        calculate_only=True)
    return out


def calculate_tonly_data(dat: DatHDF, rows, csq_mapped,
                         center_func_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Averaged Transition data for rows selected using named center func

    Args:
        dat ():
        rows ():
        csq_mapped ():
        center_func_name ():

    Returns:

    """

    def get_data(rs, csq, transition_dat: DatHDF) -> Tuple[np.ndarray, np.ndarray]:
        if csq:
            name = 'csq_mapped'
            data_group_name = 'Data'
        else:
            name = 'i_sense'
            data_group_name = 'Transition'
        s, f = rs
        x_ = transition_dat.Data.get_data('x', data_group_name=data_group_name)
        data_ = transition_dat.Data.get_data(name, data_group_name=data_group_name)[s:f]
        return x_, data_

    x, data = get_data(rows, csq_mapped, dat)
    centers = get_centers(dat, center_func_name=center_func_name, rows=rows, se_data=False)

    data_avg, x_avg = U.mean_data(x=x, data=data, centers=centers, method='linear', return_x=True)
    return x_avg, data_avg


@dataclass
class TransitionCalcParams:
    initial_x: np.ndarray  # For getting param estimates
    initial_data: np.ndarray  # For getting param estimates (1D)
    force_theta: Optional[float]
    force_gamma: Optional[float]
    csq_mapped: bool = False


def get_centers(dat: DatHDF, center_func_name: str, rows: Tuple[Optional[float], Optional[float]],
                se_data: bool = False) -> np.ndarray:
    """

    Args:
        dat (): For getting data from (and potentially saving new fits to if check_exists = False)
        center_func_name (): which transition func as string
        rows (): For rows between
        se_data (): Use SE fits instead of Transition fits (i.e. dat.SquareEntropy vs dat.Transition)

    Returns:
        array of centers
    """

    def get_fit_name(f_name: str) -> str:
        return 'centering_' + f_name

    fit_name = get_fit_name(center_func_name)
    rows = (rows[0] if rows[0] else 0, rows[1] if rows[1] else dat.Data.get_data('y').shape[0])

    if se_data:
        row_fits = [dat.SquareEntropy.get_fit(which_fit='transition', which='row', row=r,
                                              fit_name=fit_name) for r in range(*rows)]
    else:
        row_fits = [dat.Transition.get_fit(which='row', row=r, name=fit_name) for r in range(*rows)]

    return centers_from_fits(row_fits)


def set_centers(dat: DatHDF, center_func_name: str, calc_params: Optional[TransitionCalcParams] = None,
                se_data: bool = False, csq_mapped: bool = False) -> np.ndarray:
    """

    Args:
        dat (): For getting data from (and potentially saving new fits to if check_exists = False)
        center_func_name (): which transition func as string
        se_data (): Use SE fits instead of Transition fits (i.e. dat.SquareEntropy vs dat.Transition)
        csq_mapped: Whether to fit regular or csq_mapped data
        calc_params (): Used only if check_exists = False

    Returns:
        array of centers
    """

    def get_fit_name(f_name: str) -> str:
        return 'centering_' + f_name

    def get_data(d: DatHDF, csq):
        if csq:
            data_ = d.Data.get_data('csq_mapped')
        else:
            data_ = d.Data.get_data('i_sense')
        return data_

    fit_name = get_fit_name(center_func_name)

    cp = calc_params
    fit_func, params = _get_transition_fit_func_params(x=cp.initial_x, data=cp.initial_data,
                                                       t_func_name=center_func_name,
                                                       theta=cp.force_theta, gamma=cp.force_gamma)

    x = dat.Data.get_data('x')
    data = get_data(dat, csq_mapped)
    if se_data:
        name = 'csq_mapped cycled only' if csq_mapped else 'i_sense cycled only'
        inputs = dat.SquareEntropy.get_Inputs(x_array=x, i_sense=data)
        # Calculate row only output if necessary (overwrite = False)
        pre_out = dat.SquareEntropy.get_row_only_output(name=name, inputs=inputs, process_params=None,
                                                        check_exists=False, overwrite=False)
        row_fits = [dat.SquareEntropy.get_fit(which_fit='transition', which='row', row=i,
                                              x=pre_out.x,
                                              data=d,
                                              fit_name=fit_name,
                                              check_exists=False,
                                              initial_params=params, fit_func=fit_func,
                                              transition_part='cold') for i, d in enumerate(pre_out.cycled)]
    else:
        row_fits = [dat.Transition.get_fit(which='row', row=i, name=fit_name,
                                           check_exists=False,
                                           x=x,
                                           data=d,
                                           initial_params=params, fit_func=fit_func) for i, d in enumerate(data)]

    return centers_from_fits(row_fits)


def get_default_transition_params(func_name: str,
                                  x: Optional[np.ndarray] = None, data: Optional[np.ndarray] = None) -> lm.Parameters:
    params = get_param_estimates(x=x, data=data)
    if func_name == 'i_sense_digamma':
        params.add('g', 0, min=-50, max=10000, vary=True)
    elif func_name == 'i_sense_digamma_amplin':
        params.add('g', 0, min=-50, max=10000, vary=True)
        params.add('amplin', 0, vary=True)
    return params


def calculate_transition_only_fit(datnum, save_name, t_func_name: str = 'i_sense_digamma', theta=None, gamma=None,
                                  x: Optional[np.ndarray] = None, data: Optional[np.ndarray] = None,
                                  width: Optional[float] = None, center: Optional[float] = None,
                                  experiment_name: Optional[str] = None,
                                  overwrite=False) -> FitInfo:
    from src.dat_object.make_dat import get_dat
    dat = get_dat(datnum, exp2hdf=experiment_name)

    x = x if x is not None else dat.Transition.avg_x
    data = data if data is not None else dat.Transition.avg_data

    x, data = get_data_in_range(x, data, width, center=center)

    t_func, params = _get_transition_fit_func_params(x, data, t_func_name, theta, gamma)

    return dat.Transition.get_fit(name=save_name, fit_func=t_func,
                                  data=data, x=x, initial_params=params,
                                  check_exists=False, overwrite=overwrite)


def get_data_in_range(x: np.ndarray, data: np.ndarray, width: Optional[float], center: Optional[float] = None) -> \
        Tuple[np.ndarray, np.ndarray]:
    if center is None:
        center = 0
    if width is not None:
        x, data = np.copy(x), np.copy(data)

        start_ind, end_ind = U.get_data_index(x, [center-width, center+width], is_sorted=True)

        x[:start_ind] = np.nan
        x[end_ind+1:] = np.nan

        data[:start_ind] = np.nan
        data[end_ind+1:] = np.nan
    return x, data


def _get_transition_fit_func_params(x, data, t_func_name, theta, gamma):
    """

    Args:
        x ():
        data ():
        t_func_name ():
        theta ():
        gamma ():

    Returns:

    """
    t_func = get_transition_function(t_func_name)
    params = get_default_transition_params(t_func_name, x, data)
    if theta:
        params = U.edit_params(params, 'theta', value=theta, vary=False)
    if gamma is not None and 'g' in params:
        params = U.edit_params(params, 'g', gamma, False)
    return t_func, params


def calculate_se_transition(datnum: int, save_name: str, se_output_name: str, t_func_name: str = 'i_sense_digamma',
                            theta=None, gamma=None,
                            transition_part: str = 'cold',
                            width: Optional[float] = None, center: Optional[float] = None,
                            experiment_name: Optional[str] = None,
                            overwrite=False):
    from src.dat_object.make_dat import get_dat
    dat = get_dat(datnum, exp2hdf=experiment_name)
    data = dat.SquareEntropy.get_transition_part(name=se_output_name, part=transition_part, existing_only=True)
    x = dat.SquareEntropy.get_Outputs(name=se_output_name, check_exists=True).x

    x, data = get_data_in_range(x, data, width, center=center)

    t_func, params = _get_transition_fit_func_params(x, data, t_func_name, theta, gamma)

    return dat.SquareEntropy.get_fit(which_fit='transition', transition_part=transition_part, fit_name=save_name,
                                     fit_func=t_func, initial_params=params, data=data, x=x, check_exists=False,
                                     overwrite=overwrite)


def calculate_se_entropy_fit(datnum: int, save_name: str, se_output_name: str,
                             width: Optional[float] = None, center: Optional[float] = None,
                             experiment_name: Optional[str] = None,
                             overwrite=False):
    from src.dat_object.make_dat import get_dat
    dat = get_dat(datnum, exp2hdf=experiment_name)
    out = dat.SquareEntropy.get_Outputs(name=se_output_name, check_exists=True)
    x = out.x
    data = out.average_entropy_signal

    x, data = get_data_in_range(x, data, width, center)
    return dat.Entropy.get_fit(name=save_name, x=out.x, data=data, check_exists=False, overwrite=overwrite)


