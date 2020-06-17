from typing import Union, NamedTuple, List
import src.Configs.Main_Config as cfg
import src.CoreUtil as CU
from src.CoreUtil import data_to_NamedTuple
from src.DatHDF import Util as DHU
import abc
import h5py
import logging
import numpy as np
import lmfit as lm


logger = logging.getLogger(__name__)


class DatAttribute(abc.ABC):
    version = 'NEED TO OVERRIDE'
    group_name = 'NEED TO OVERRIDE'

    def __init__(self, hdf):
        self.version = self.__class__.version
        self.hdf = hdf
        self.group = self.get_group()
        self._set_default_group_attrs()


    def get_group(self):
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
        e.g. upon creation of new dat, add description of Entropy group in attrs"""
        if 'version' not in self.group.attrs.keys():
            self.group.attrs['version'] = self.__class__.version

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
        self.all_fits = None  # type: Union[List[DHU.FitInfo], None]
        self.avg_fit = None  # type: Union[DHU.FitInfo, None]

        self.get_from_HDF()

    @abc.abstractmethod
    def get_from_HDF(self):
        """Should be able to run this to get all data from HDF into expected attrs of FittingAttr
        below is just getting started: self.x/y/avg_fit/all_fits"""
        dg = self.group['Data']
        self.x = dg.get('x', None)
        self.y = dg.get('y', None)
        if isinstance(self.y, float) and np.isnan(self.y):  # Because I store None as np.nan
            self.y = None

        avg_fit_group = self.group.get('Avg_fit', None)
        if avg_fit_group is not None:
            self.avg_fit = DHU.fit_group_to_FitInfo(avg_fit_group)

        row_fits_group = self.group.get('Row_fits', None)
        if row_fits_group is not None:
            self.all_fits = DHU.rows_group_to_all_FitInfos(row_fits_group)
        # Init rest here (self.data, self.avg_data, self.avg_data_err...)

    @abc.abstractmethod
    def update_HDF(self):
        """Should be able to run this to set all data in HDF s.t. loading would return to current state"""
        super().update_HDF()
        self._set_data_hdf()
        self._set_row_fits_hdf()
        self._set_avg_data_hdf()
        self._set_avg_fit_hdf()

    @abc.abstractmethod
    def _set_data_hdf(self, data_name=None):
        """Set non-averaged data in HDF (x, y, data)"""
        data_name = data_name if data_name is not None else 'data'
        tdg = self.group.require_group('Data')
        for name, data in zip(['x', 'y', data_name], [self.x, self.y, self.data]):
            if data is None:
                data = np.nan
            tdg[name] = data

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
            fit_infos = [DHU.FitInfo() for _ in row_fits]
            for fi, rf in zip(fit_infos, row_fits):
                fi.init_from_fit(rf)
            self.all_fits = fit_infos
            self._set_row_fits_hdf()

    @abc.abstractmethod
    def _set_row_fits_hdf(self):
        """Save fit_info per row to HDF"""
        row_fits_group = self.group.require_group('Row_fits')
        y = self.y[:]
        if y is None:
            y = [None]*len(self.all_fits)
        for i, (fit_info, y_val) in enumerate(zip(self.all_fits, y)):
            name = f'Row{i}:{y_val:.1g}' if y_val is not None else f'Row{i}'
            row_group = row_fits_group.require_group(name)
            row_group.attrs['row'] = i  # Used when rebuilding to make sure things are in order
            row_group.attrs['y_val'] = y_val if y_val is not None else np.nan
            fit_info.save_to_hdf(row_group)

    @abc.abstractmethod
    def set_avg_data(self, center_ids):
        """Make average data by centering rows of self.data with center_ids then averaging then save to HDF"""
        assert self.all_fits is not None
        assert self.data.ndim == 2
        if center_ids is None:
            logger.warning(f'Averaging data with no center IDs')
            center_ids = np.zeros(shape=self.data.shape[0])
        self.avg_data, self.avg_data_err = CU.average_data(self.data, center_ids)
        self._set_avg_data_hdf()

    @abc.abstractmethod
    def _set_avg_data_hdf(self):
        """Save average data to HDF"""
        dg = self.group['Data']
        # dg['avg_i_sense'] = self.avg_data
        # dg['avg_i_sense_err'] = self.avg_data_err

    @abc.abstractmethod
    def run_avg_fit(self, fitter=None, params=None, auto_bin=True):
        """Run fit on average data"""
        if self.avg_data is None:
            logger.info('self.avg_data was none, running set_avg_data first')
            self.set_avg_data(center_ids=None)
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
            fit_info = DHU.FitInfo()
            fit_info.init_from_fit(fit)
            self.avg_fit = fit_info
            self._set_avg_fit_hdf()

    @abc.abstractmethod
    def _set_avg_fit_hdf(self):
        """Save average fit to HDF"""
        avg_fit_group = self.group.require_group('Avg_fit')
        self.avg_fit.save_to_hdf(avg_fit_group)


##################################

def get_instr_vals(instr: str, instrid: Union[int, str, None], infodict) -> Union[NamedTuple, None]:
    instrname, instr_tuple = get_key_ntuple(instr, instrid)
    logs = infodict.get('Logs', None)
    if logs is not None:
        try:
            if instrname in logs.keys():
                instrinfo = logs[instrname]
            elif instr+'s' in logs.keys() and logs[instr+'s'] is not None and instrname in logs[instr+'s'].keys():
                instrinfo = logs[instr+'s'][instrname]
            else:
                return None
            if instrinfo is not None:
                ntuple = data_to_NamedTuple(instrinfo, instr_tuple)  # Will leave warning in cfg.warning if necessary
            else:
                return None
            if cfg.warning is not None:
                logger.warning(f'For {instrname} - {cfg.warning}')
        except (TypeError, KeyError):
            logger.info(f'No {instr} found')
            return None
        return ntuple
    return None


def get_key_ntuple(instrname: str, instrid: Union[str, int] = None) -> [str, NamedTuple]:
    """Returns instrument key and namedtuple for that instrument"""
    instrtupledict = {'srs': SRStuple, 'mag': MAGtuple, 'temperatures': TEMPtuple}
    if instrname not in instrtupledict.keys():
        raise KeyError(f'No {instrname} found')
    else:
        if instrid is None:
            instrid = ''
        instrkey = instrname + str(instrid)
    return instrkey, instrtupledict[instrname]


#  name in Logs dict has to be exactly the same as NamedTuple attr names
class SRStuple(NamedTuple):
    gpib: int
    out: int
    tc: float
    freq: float
    phase: float
    sens: float
    harm: int
    CH1readout: int


class MAGtuple(NamedTuple):
    field: float
    rate: float


class TEMPtuple(NamedTuple):
    mc: float
    still: float
    mag: float
    fourk: float
    fiftyk: float

