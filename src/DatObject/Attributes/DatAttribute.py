import inspect
from typing import Union, List
from src import CoreUtil as CU, Main_Config as cfg
import abc
import h5py
import logging
import numpy as np
import lmfit as lm

from src.HDF_Util import params_from_HDF, params_to_HDF
import src.HDF_Util as HDU

from src.DatObject.DatHDF import DatHDF

logger = logging.getLogger(__name__)


class DatAttribute(abc.ABC):
    version = 'NEED TO OVERRIDE'
    group_name = 'NEED TO OVERRIDE'

    def __init__(self, hdf):
        self.version = self.__class__.version
        self.hdf = hdf
        self.group = self._get_group()
        self._set_default_group_attrs()

    def _get_group(self):
        """Sets self.group to be the appropriate group in HDF for given DatAttr
        based on the class.group_name which should be overridden.
        Will create group in HDF if necessary"""
        group_name = self.__class__.group_name
        if group_name not in self.hdf.keys():
            self.hdf.create_group(group_name)
        group = self.hdf[group_name]  # type: h5py.Group
        return group

    @abc.abstractmethod
    def _set_default_group_attrs(self):
        """Set default attributes of group if not already existing
        e.g. upon creation of new dat, add description of group in attrs"""
        if 'version' not in self.group.attrs.keys():
            self.group.attrs['version'] = self.__class__.version
        if 'description' not in self.group.attrs.keys():
            self.group.attrs['description'] = "This should be overwritten in subclass!"

    @abc.abstractmethod
    def get_from_HDF(self):
        """Should be able to run this to get all data from HDF into expected attrs of DatAttr"""
        pass

    @abc.abstractmethod
    def update_HDF(self):
        """Should be able to run this to set all data in HDF s.t. loading would return to current state"""
        self.group.attrs['version'] = self.__class__.version


class FittingAttribute(DatAttribute, abc.ABC):
    def __init__(self, hdf):
        super().__init__(hdf)
        self.x = None
        self.y = None
        self.data = None
        self.avg_data = None
        self.avg_data_err = None
        self.fit_func = None
        self.all_fits: Union[List[FitInfo], None] = None
        self.avg_fit: Union[FitInfo, None] = None

        self.get_from_HDF()

    @abc.abstractmethod
    def get_from_HDF(self):
        """Should be able to run this to get all data from HDF into expected attrs of FittingAttr
        below is just getting started: self.x/y/avg_fit/all_fits"""

        dg = self.group.get('Data', None)
        if dg is not None:
            self.x = dg.get('x', None)
            self.y = dg.get('y', None)
            if isinstance(self.y, float) and np.isnan(self.y):  # Because I store None as np.nan
                self.y = None

        avg_fit_group = self.group.get('Avg_fit', None)
        if avg_fit_group is not None:
            self.avg_fit = fit_group_to_FitInfo(avg_fit_group)

        row_fits_group = self.group.get('Row_fits', None)
        if row_fits_group is not None:
            self.all_fits = rows_group_to_all_FitInfos(row_fits_group)
        # Init rest here (self.data, self.avg_data, self.avg_data_err...)

    @abc.abstractmethod
    def update_HDF(self):
        """Should be able to run this to set all data in HDF s.t. loading would return to current state"""
        super().update_HDF()
        self._set_data_hdf()
        self._set_row_fits_hdf()
        self._set_avg_data_hdf()
        self._set_avg_fit_hdf()
        self.hdf.flush()

    @abc.abstractmethod
    def _set_data_hdf(self, data_name=None):
        """Set non-averaged data in HDF (x, y, data)"""
        data_name = data_name if data_name is not None else 'data'
        dg = self.group.require_group('Data')
        for name, data in zip(['x', 'y', data_name], [self.x, self.y, self.data]):
            if data is None:
                data = np.nan
            HDU.set_data(dg, name, data)  # Removes dataset before setting if necessary
        self.hdf.flush()

    @abc.abstractmethod
    def run_row_fits(self, fitter=None, params=None, auto_bin=True):
        """Run fits per row"""
        assert all([data is not None for data in [self.x, self.data]])
        if params is None:
            if hasattr(self.avg_fit, 'params'):
                params = self.avg_fit.params
            else:
                params = None
        if fitter is None:
            return params  # Can use up to here by calling super().run_row_fits(params=params)

        elif fitter is not None:  # Otherwise implement something like this in override
            x = self.x[:]
            data = self.data[:]
            row_fits = fitter(x, data, params, auto_bin=auto_bin)  # type: List[lm.model.ModelResult]
            fit_infos = [FitInfo() for _ in row_fits]
            for fi, rf in zip(fit_infos, row_fits):
                fi.init_from_fit(rf)
            self.all_fits = fit_infos
            self._set_row_fits_hdf()

    @abc.abstractmethod
    def _set_row_fits_hdf(self):
        """Save fit_info per row to HDF"""
        if self.all_fits is not None:
            row_fits_group = self.group.require_group('Row_fits')
            if self.y.shape != ():
                y = self.y[:]
            else:
                y = None
            row_fits_to_group(row_fits_group, self.all_fits, y)
            self.hdf.flush()

    @abc.abstractmethod
    def set_avg_data(self, centers, x_array=None):
        """Make average data by centering rows of self.data with centers (defined on original x_array or x_array)
         then averaging then save to HDF

        Args:
            centers (Union[np.ndarray, str]): Center positions defined on x_array or original x_array by default
            x_array (np.ndarray): Optional x_array which centers were defined on

        Returns:
            None: Sets self.avg_data, self.avg_data_err and saves to HDF
        """
        x = x_array if x_array is not None else self.x
        if self.data.ndim == 1:
            self.avg_data = self.data
            self.avg_data_err = np.nan
        else:
            if centers is None:
                logger.warning(f'Averaging data with no centers passed')
                centered_data = self.data
            elif centers == 'None':  # Explicit no centering, so no need for warning
                centered_data = self.data
            else:
                centered_data = CU.center_data(x, self.data, centers)
            self.avg_data = np.nanmean(centered_data, axis=0)
            self.avg_data_err = np.nanstd(centered_data, axis=0)
        self._set_avg_data_hdf()

    @abc.abstractmethod
    def _set_avg_data_hdf(self):
        """Save average data to HDF"""
        dg = self.group['Data']
        self.hdf.flush()
        # dg['avg_i_sense'] = self.avg_data
        # dg['avg_i_sense_err'] = self.avg_data_err

    @abc.abstractmethod
    def run_avg_fit(self, fitter=None, params=None, auto_bin=True):
        """Run fit on average data"""
        if self.avg_data is None:
            logger.info('self.avg_data was none, running set_avg_data first')
            self.set_avg_data(centers=None)
        assert all([data is not None for data in [self.x, self.avg_data]])

        if params is None:
            if hasattr(self.avg_fit, 'params'):
                params = self.avg_fit.params
            else:
                params = None

        if fitter is None:
            return params  # Can use up to here by calling super().run_row_fits(params=params)

        elif fitter is not None:  # Otherwise implement something like this in override
            x = self.x[:]
            data = self.avg_data[:]
            fit = fitter(x, data, params, auto_bin=auto_bin)[0]  # Note: Expecting to returned a list of 1 fit.
            fit_info = FitInfo()
            fit_info.init_from_fit(fit)
            self.avg_fit = fit_info
            self._set_avg_fit_hdf()
        else:
            raise NotImplementedError

    @abc.abstractmethod
    def _set_avg_fit_hdf(self):
        """Save average fit to HDF"""
        if self.avg_fit is not None:
            avg_fit_group = self.group.require_group('Avg_fit')
            self.avg_fit.save_to_hdf(avg_fit_group)
            self.hdf.flush()


class Values(object):
    """Object to store Init/Best values in and stores Keys of those values in self.keys"""
    def __getattr__(self, item):
        if item.startswith('__') or item.startswith('_') or item == 'keys':  # So don't complain about things like __len__
            return super().__getattribute__(item)  # Come's here looking for Ipython variables
        else:
            if item in self.keys:
                return super().__getattribute__(item)  # TODO: same as above, does this ever get called?
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
        return val

    def __setattr__(self, key, value):
        if key.startswith('__') or key.startswith('_') or key == 'keys' or not isinstance(value, (np.number, float, int, type(None))):  # So don't complain about
            # things like __len__ and don't keep key of random things attached to class
            super().__setattr__(key, value)
        else:  # probably is something I want the key of
            self.keys.append(key)
            super().__setattr__(key, value)

    def __repr__(self):
        string = ''
        for key in self.keys:
            string += f'{key}={self.__getattr__(key):.5g}\n'
        return string

    def __init__(self):
        self.keys = []


class FitInfo(object):
    def __init__(self):
        self.params: Union[lm.Parameters, None] = None
        self.func_name: Union[str, None] = None
        self.func_code: Union[str, None] = None
        self.fit_report: Union[str, None] = None
        self.model: Union[lm.Model, None] = None
        self.best_values: Union[Values, None] = None
        self.init_values: Union[Values, None] = None
        # Will only exist when set from fit, or after recalculate_fit
        self.fit_result: Union[lm.model.ModelResult, None] = None

    def init_from_fit(self, fit: lm.model.ModelResult):
        """Init values from fit result"""
        if fit is None:
            logger.warning(f'Got None for fit to initialize from. Not doing anything.')
            return None
        assert isinstance(fit, lm.model.ModelResult)
        self.params = fit.params
        self.func_name = fit.model.func.__name__

        #  Can't get source code when running from deepcopy (and maybe other things will break this)
        try:
            func_code = inspect.getsource(fit.model.func)
        except OSError:
            if self.func_code is not None:
                func_code = '[WARNING: might not be correct as fit was re run and could not get source code'+self.func_code
            else:
                logger.warning('Failed to get source func_code and no existing func_code')
                func_code = 'Failed to get source code due to OSError'
        self.func_code = func_code

        self.fit_report = fit.fit_report()
        self.model = fit.model
        self.best_values = Values()
        self.init_values = Values()
        for key in self.params.keys():
            par = self.params[key]
            self.best_values.__setattr__(par.name, par.value)
            self.init_values.__setattr__(par.name, par.init_value)

        self.fit_result = fit

    def init_from_hdf(self, group: h5py.Group):
        """Init values from HDF file"""
        self.params = params_from_HDF(group)
        self.func_name = group.attrs.get('func_name', None)
        self.func_code = group.attrs.get('func_code', None)
        self.fit_report = group.attrs.get('fit_report', None)
        self.model = lm.models.Model(self._get_func())
        self.best_values = Values()
        self.init_values = Values()
        for key in self.params.keys():
            par = self.params[key]
            self.best_values.__setattr__(par.name, par.value)
            self.init_values.__setattr__(par.name, par.init_value)

        self.fit_result = None

    def save_to_hdf(self, group: h5py.Group):
        if self.params is None:
            logger.warning(f'No params to save for {self.func_name} fit. Not doing anything')
            return None
        params_to_HDF(self.params, group)
        group.attrs['description'] = 'FitInfo'  # Overwrites what params_to_HDF sets
        group.attrs['func_name'] = self.func_name
        group.attrs['func_code'] = self.func_code
        group.attrs['fit_report'] = self.fit_report
        group.file.flush()

    def _get_func(self):
        """Cheeky way to get the function which was used for fitting (stored as text in HDF so can be executed here)
        Definitely not ideal, so I at least check that I'm not overwriting something, but still should be careful here"""
        return HDU.get_func(self.func_name, self.func_code)

    def eval_fit(self, x: np.ndarray):
        """Return best fit for x array using params"""
        return self.model.eval(self.params, x=x)

    def eval_init(self, x: np.ndarray):
        """Return init fit for x array using params"""
        init_pars = CU.edit_params(self.params, list(self.params.keys()), [par.init_value for par in self.params.values()])
        return self.model.eval(init_pars, x=x)

    def recalculate_fit(self, x: np.ndarray, data: np.ndarray, auto_bin=False):
        """Fit to data with x array and update self"""
        assert data.ndim == 1
        data, x = CU.remove_nans(data, x)
        if auto_bin is True and len(data) > cfg.FIT_NUM_BINS:
            logger.info(f'Binning data of len {len(data)} into {cfg.FIT_NUM_BINS} before fitting')
            x, data = CU.bin_data([x, data], round(len(data)/cfg.FIT_NUM_BINS))
        fit = self.model.fit(data.astype(np.float32), self.params, x=x, nan_policy='omit')
        self.init_from_fit(fit)

    def edit_params(self, param_names=None, values=None, varys=None, mins=None, maxs=None):
        self.params = CU.edit_params(self.params, param_names, values, varys, mins, maxs)

    @classmethod
    def from_fit(cls, fit):
        inst = cls()
        inst.init_from_fit(fit)
        return inst


def row_fits_to_group(group, fits, y_array=None):
    """For saving all row fits in a dat in a group. To get back to original, use rows_group_to_all_FitInfos"""
    if y_array is None:
        y_array = [None] * len(fits)
    else:
        assert len(y_array) == len(fits)
    for i, (fit_info, y_val) in enumerate(zip(fits, y_array)):
        name = f'Row{i}:{y_val:.5g}' if y_val is not None else f'Row{i}'
        row_group = group.require_group(name)
        row_group.attrs['row'] = i  # Used when rebuilding to make sure things are in order
        row_group.attrs['y_val'] = y_val if y_val is not None else np.nan
        fit_info.save_to_hdf(row_group)


def rows_group_to_all_FitInfos(group: h5py.Group):
    """For loading row fits saved with row_fits_to_group"""
    row_group_dict = {}
    for key in group.keys():
        row_id = group[key].attrs.get('row', None)
        if row_id is not None and group[key].attrs.get('description', None) == "FitInfo":
            row_group_dict[row_id] = group[key]
    fit_infos = [FitInfo() for _ in row_group_dict]  # Makes a new FitInfo() [FI()]*10 just gives 10 pointers to 1 obj
    for key in sorted(row_group_dict.keys()):
        fit_infos[key].init_from_hdf(row_group_dict[key])
    return fit_infos


def fit_group_to_FitInfo(group: h5py.Group):
    """For loading a single Fit group from HDF (i.e. if saved using FitInfo.save_to_hdf()"""
    assert group.attrs.get('description', None) in ["FitInfo", 'Single Parameters of fit']
    fit_info = FitInfo()
    fit_info.init_from_hdf(group)
    return fit_info


##################################

# def get_instr_vals(instr: str, instrid: Union[int, str, None], infodict) -> Union[NamedTuple, None]:
#     instrname, instr_tuple = get_key_ntuple(instr, instrid)
#     logs = infodict.get('Logs', None)
#     if logs is not None:
#         try:
#             if instrname in logs.keys():
#                 instrinfo = logs[instrname]
#             elif instr+'s' in logs.keys() and logs[instr+'s'] is not None and instrname in logs[instr+'s'].keys():
#                 instrinfo = logs[instr+'s'][instrname]
#             else:
#                 return None
#             if instrinfo is not None:
#                 ntuple = data_to_NamedTuple(instrinfo, instr_tuple)  # Will leave warning in cfg.warning if necessary
#             else:
#                 return None
#             if cfg.warning is not None:
#                 logger.warning(f'For {instrname} - {cfg.warning}')
#         except (TypeError, KeyError):
#             logger.info(f'No {instr} found')
#             return None
#         return ntuple
#     return None
#
#
# def get_key_ntuple(instrname: str, instrid: Union[str, int] = None) -> [str, NamedTuple]:
#     """Returns instrument key and namedtuple for that instrument"""
#     instrtupledict = {'srs': SRStuple, 'mag': MAGtuple, 'temperatures': TEMPtuple}
#     if instrname not in instrtupledict.keys():
#         raise KeyError(f'No {instrname} found')
#     else:
#         if instrid is None:
#             instrid = ''
#         instrkey = instrname + str(instrid)
#     return instrkey, instrtupledict[instrname]
#
#
# #  name in Logs dict has to be exactly the same as NamedTuple attr names



