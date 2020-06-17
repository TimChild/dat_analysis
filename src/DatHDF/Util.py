import src.Configs.Main_Config as cfg
from src import CoreUtil as CU
import os
import h5py
import numpy as np
import lmfit as lm
import inspect
import logging

logger = logging.getLogger(__name__)


def get_dat_hdf_path(dat_id, dir=None, overwrite=False):
    dir = dir if dir else CU.get_full_path(cfg.hdfdir)
    file_path = os.path.join(dir, dat_id+'.h5')
    if os.path.exists(file_path) and overwrite is True:
        os.remove(file_path)
    if not os.path.exists(file_path):  # make empty file then return path
        dir, _ = os.path.split(file_path)
        os.makedirs(dir, exist_ok=True)  # Ensure directory exists
        f = h5py.File(file_path, 'w')  # Init a HDF file
        f.close()
    return file_path


class Values(object):
    """Object to store Init/Best values in and stores Keys of those values in self.keys"""
    def __getattr__(self, item):
        if item.startswith('__') or item.startswith('_'):  # So don't complain about things like __len__
            return super().__getattribute__(self, item)
        else:
            if item in self.keys:
                return super().__getattribute__(self, item)
            else:
                msg = f'{item} does not exist. Valid keys are {self.keys}'
                print(msg)
                logger.warning(msg)
                return None

    def __setattr__(self, key, value):
        if key.startswith('__') or key.startswith('_') or not isinstance(value, (float, int, type(None))):  # So don't complain about
            # things like __len__ and don't keep key of random things attached to class
            super().__setattr__(self, key, value)
        else:  # probably is something I want the key of
            self.keys.append(key)
            super().__setattr__(key, value)

    def __repr__(self):
        for key in self.keys:
            print(f'{key}={self.__getattr__(key)}\n')

    def __init__(self):
        self.keys = []


class FitInfo(object):
    def __init__(self):
        self.params = None  # type: lm.Parameters
        self.func_name = None  # type: str
        self.func_code = None  # type: str
        self.fit_report = None  # type: str
        self.model = None  # type: lm.Model
        self.best_values = None  # type: Values
        self.init_values = None  # type: Values
        # Will only exist when set from fit, or after recalculate_fit
        self.fit_result = None  # type: lm.model.ModelResult

    def init_from_fit(self, fit: lm.model.ModelResult):
        """Init values from fit result"""
        self.params = fit.params
        self.func_name = fit.model.func.__name__
        self.func_code = inspect.getsource(fit.model.func)
        self.fit_report = fit.fit_report()
        self.model = fit.model
        self.best_values = Values()
        self.init_values = Values()
        for par in self.params:
            self.best_values.__setattr__(par['name'], par.value)
            self.init_values.__setattr__(par['name'], par.init_value)

        self.fit_result = fit

    def init_from_hdf(self, group: h5py.Group):
        """Init values from HDF file"""
        self.params = _params_from_HDF(group)
        self.func_name = group.attrs.get('func_name', None)
        self.func_code = group.attrs.get('func_code', None)
        self.fit_report = group.attrs.get('fit_report', None)
        self.model = lm.models.Model(self._get_func())
        self.best_values = Values()
        self.init_values = Values()
        for par in self.params:
            self.best_values.__setattr__(par['name'], par.value)
            self.init_values.__setattr__(par['name'], par.init_value)

        self.fit_result = None
        pass

    def save_to_hdf(self, group: h5py.Group):
        assert self.params is not None
        _params_to_HDF(self.params, group)
        group.attrs['func_name'] = self.func_name
        group.attrs['func_code'] = self.func_code
        group.attrs['fit_report'] = self.fit_report

    def _get_func(self):
        """Cheeky way to get the function which was used for fitting (stored as text in HDF so can be executed here)
        Definitely not ideal, so I at least check that I'm not overwriting something, but still should be careful here"""
        if self.func_name not in globals().keys():
            logger.info(f'Executing: {self.func_code}')
            exec(self.func_code)  # Should be careful about this! Just running whatever code is stored in HDF
        else:
            logger.info(f'Func {self.func_name} already exists so not running self.func_code')
        func = globals()[self.func_name]  # Should find the function which already exists or was executed above
        assert callable(func)
        return func

    def eval_fit(self, x: np.ndarray):
        """Return best fit for x array using params"""
        return self.model.eval(self.params, x=x)

    def eval_init(self, x: np.ndarray):
        """Return init fit for x array using params"""
        init_pars = CU.edit_params(self.params, [self.params.keys()], [par.init_value for par in self.params])
        return self.model.eval(init_pars, x=x)

    def recalculate_fit(self, x: np.ndarray, data: np.ndarray, auto_bin=False):
        """Fit to data with x array and update self"""
        assert data.ndim == 1
        data, x = CU.remove_nans(data, x)
        if auto_bin is True and len(data) > cfg.FIT_BINSIZE:
            logger.info(f'Binning data of len {len(data)} into {cfg.FIT_BINSIZE} before fitting')
            x, data = CU.bin_data([x, data], cfg.FIT_BINSIZE)
        fit = self.model.fit(data.astype(np.float32), self.params, x=x)
        self.init_from_fit(fit)



PARAM_KEYS = ['name', 'value', 'vary', 'min', 'max', 'expr', 'brute_step']


def _params_to_HDF(params: lm.Parameters, group: h5py.Group):
    group.attrs['description'] = "Single Parameters of fit"
    for key in params.keys():
        par = params[key]
        par_group = group.require_group(key)
        par_group.attrs['description'] = "Single Param"
        for par_key in PARAM_KEYS:
            attr_val = getattr(par, par_key, np.nan)
            attr_val = attr_val if attr_val is not None else np.nan
            par_group.attrs[par_key] = attr_val
        par_group.attrs['init_value'] = getattr(par, 'init_value', np.nan)
        par_group.attrs['stderr'] = getattr(par, 'stderr', np.nan)
    pass


def _params_from_HDF(group) -> lm.Parameters:
    params = lm.Parameters()
    for key in group.keys():
        if isinstance(group[key], h5py.Group) and group[key].attrs.get('description', None) == 'Single Param':
            par_group = group[key]
            par_vals = [par_group.attrs.get(par_key, None) for par_key in PARAM_KEYS]
            par_vals = [v if not (isinstance(v, float) and np.isnan(v)) else None for v in par_vals]
            params.add(*par_vals)  # create par
            par = params[key]  # Get single par
            par.stderr = par_group.attrs.get('stderr', None)
            par.value = par.init_value  # Because the saved value was actually final value, but inits into init_val
            par.init_value = par_group.attrs.get('init_value', None)  # I save init_value separately
            for par_key in PARAM_KEYS+['stderr', 'init_value']:
                if getattr(par, par_key) == np.nan:  # How I store None in HDF
                    setattr(par, par_key, None)
    return params


def rows_group_to_all_FitInfos(group:h5py.Group):
    row_group_dict = {}
    for key in group.keys():
        row_id = group[key].attrs.get('row', None)
        if row_id is not None and group[key].attrs.get('description', None) == "Single Parameters of fit":
            row_group_dict[row_id] = group[key]
    fit_infos = [DHU.FitInfo()]*len(row_group_dict)
    for key in sorted(row_group_dict.keys()):
        fit_infos[key].init_from_hdf(row_group_dict[key])
    return fit_infos


def fit_group_to_FitInfo(group:h5py.Group):
    assert group.attrs.get('description', None) == "Single Parameters of fit"
    fit_info = DHU.FitInfo()
    fit_info.init_from_hdf(group)
    return fit_info