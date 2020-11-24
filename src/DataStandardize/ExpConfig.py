"""
For making ExpConfig group in DatHDF

Idea of ExpConfig group is to store any additional information which is not in ExpDat directly in a way that allows
DatAttrs to initialize properly.

e.g. Dat.Data should store 'i_sense' data in nA, but the data recorded might have been 10x smaller and called
'cscurrent', so there needs to be something somewhere which says that the 'cscurrent' is the 'i_sense' data,
but it needs to be multiplied by 10 first

Also can store info in here like voltage divider set ups which otherwise aren't necessarily stored in Exp data

ExpConfigBase should be subclassed for each new cooldown in general to override whatever needs to be changed for that
exp directly. For one off changes, modifying the DatHDFs directly is also a good options, just be careful not to
overwrite it as the info to reinitialize properly won't be there """
from __future__ import annotations
import abc
import src.HDF_Util as HDU
from src.DatObject.Attributes.Logs import replace_in_json
from src.DatObject.Attributes.DatAttribute import DatAttribute, DataDescriptor, DatDataclassTemplate
from typing import TYPE_CHECKING, Dict, Union, List
from src.HDF_Util import with_hdf_read, with_hdf_write, NotFoundInHdfError
from functools import wraps, lru_cache
from dataclasses import dataclass
from src import CoreUtil as CU
from dictor import dictor
import logging
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF

from src import CoreUtil as CU


@dataclass
class DataInfo(DatDataclassTemplate):
    standard_name: str
    offset: float = 0.0
    multiply: float = 1.0


# Standard names that are used throughout code (i.e. 'x', 'y', 'i_sense')
# Full example: 'cscurrent': DataInfo('i_sense', offset=1.23e-6, multiply=1e8)
X_DATA = DataInfo('x')
Y_DATA = DataInfo('y')
I_SENSE_DATA = DataInfo('i_sense')


class ExpConfigBase(abc.ABC):
    """
    Base Config class to outline what info needs to be in any exp specific config

    Should be things which are Experiment specific but not stored in the original HDF directly.

    Note: Any information in here should be somehow stored in HDF by ExpConfigDatAttribute on first init. (idea is to
    have all information related to measurements in the DatHDF)

    # TODO: Make something where this info can be stored/read from a JSON file (i.e. so easier to modify with out going to code)
    """

    def __init__(self, datnum=None):
        self.datnum = datnum

    @abc.abstractmethod
    def get_sweeplogs_json_subs(self) -> dict:
        """Something that returns a list of re match/repl strings to fix sweeplogs JSON for a given datnum.
        i.e. if some of the sweeplogs saved were not valid JSON's then JSON parse will not work until things are fixed

        Form: {match: repl, match, repl,..}

        If none needed, return None
        """
        return {'FastDAC 1': 'FastDAC'}  # Example

    @abc.abstractmethod
    def get_sweeplog_modifications(self) -> dict:
        """
        Something that returns a dict of keys to switch, remove, and/or add. Paths to values should be '.' separated.
        i.e. useful if you know that one of the keys in the sweeplogs is saved incorrectly, or it is nested too deep or
        if there should be an additional entry for a missing instrument or something.

        Note: if nothing is wrong, just return an empty dict
        Returns:
            (dict): {'switch':{<path_to_original>:<new_path>},
                    'remove':[<keys to remove>],
                    'add':[<dicts to add to base sweeplogs>]}
        """
        # Example (where Temperatures are stored in
        # sweeplogs.Lakeshore.Temperatures.... instead of sweeplogs.Temperatures)
        switch = {'Lakeshore.Temperature': 'Temperatures'}
        remove = ['Lakeshore']  # Nothing else in 'Lakeshore' after 'Temperatures' are switched out
        add = {}
        return {'switch': switch, 'remove': remove, 'add': add}

    def get_default_data_info(self) -> Dict[str, DataInfo]:
        """
        Override to return a dictionary of standard_key: DataDescriptor(s).
        i.e. This program generally expects to find 'x' (and 'y' for 2D) arrays but they may be recorded under a
        different name. Or maybe your 'i_sense' data was recorded with an known offset you'd like to correct.

        Can assign multiple DataDescriptors to one standard_key for multiple names

        Returns:
            (dict):
        """
        info = {
            'x_array': X_DATA,
            'y_array': Y_DATA,
            'cscurrent': I_SENSE_DATA,
            'cscurrent_2d': I_SENSE_DATA,
        }
        return info


def check_exp_config_present(func):
    """Decorator to check that self.exp_config is not None"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        obj = args[0]
        assert isinstance(obj, ExpConfigGroupDatAttribute)
        if obj.exp_config is None:
            raise ValueError(f'Need to set dat.ExpConfig.exp_config to an instance of '
                             f'src.DataStandardize.BaseClasses.Exp2HDF before calling this function (i.e. this '
                             f'method was intended to be called in for initialization only)')
        else:
            return func(*args, **kwargs)

    return wrapper


class ExpConfigGroupDatAttribute(DatAttribute):
    """
    Given a HDFContainer and
    """
    version = '1.0.0'
    group_name = 'Experiment Config'
    description = 'Information about the experiment which is not stored in the dat file directly. ' \
                  'e.g. potential dividers used, current amplifier settings, or anything else you want. ' \
                  'Can also include things that are required to fix the data recorded after the fact. ' \
                  'e.g. parts of sweeplogs which need to be replaced to make them valid JSONs, rows of data to avoid...'

    def __init__(self, dat: DatHDF, exp_config: ExpConfigBase = None):
        self.exp_config = exp_config
        super().__init__(dat)

    @with_hdf_write  # Things in here are quick and all require writing so just open in write mode once for all
    def initialize_minimum(self):
        self._set_sweeplog_subs()
        self._set_sweeplog_modification()
        self._set_default_data_descriptors()
        self.initialized = True

    @check_exp_config_present
    def _set_sweeplog_subs(self):
        self.set_group_attr('sweeplog_substitutions', self.exp_config.get_sweeplogs_json_subs())

    @check_exp_config_present
    def _set_sweeplog_modification(self):
        self.set_group_attr('sweeplog_modifications', self.exp_config.get_sweeplog_modifications())

    @check_exp_config_present
    @with_hdf_write
    def _set_default_data_descriptors(self):
        """Put the default DataDescriptors into the ExpConfig Group so that Data can find them there and use them"""
        descriptors_dict = self.exp_config.get_default_data_info()
        group = self.hdf.group.require_group('Default DataDescriptors')
        for k, v in descriptors_dict.items():
            v.save_to_hdf(group, name=k)  # Save with experiment data name

    @lru_cache  # Can change to just 'cache' once on Python 3.9+
    @with_hdf_read
    def get_sweeplogs(self) -> dict:
        """Something which returns good sweeplogs as one big json dict"""
        group = self.hdf.get('Experiment Copy')
        sweeplog_str = group.get('metadata').attrs.get('sweep_logs')
        if sweeplog_str:
            # Fix necessary substitutions in sweeplogs
            subs = self.get_group_attr('sweeplog_substitutions', None)
            sweeplog_dict = replace_in_json(sweeplog_str, subs)

            # Sweeplog modifications
            mods = self.get_group_attr('sweeplog_modifications', None)
            if mods:
                switch = mods.get('switch', None)
                remove = mods.get('remove', None)
                add = mods.get('add', None)
                if switch:
                    for k, v in switch.items():
                        temp = dictor(sweeplog_dict, k, None)
                        if not temp:
                            logger.error(f'{k} not found in sweeplogs for dat{self.dat.datnum}: {sweeplog_dict}')
                        else:
                            CU.nested_dict_val(sweeplog_dict, k, mode='pop')  # Remove info from old location
                            CU.nested_dict_val(sweeplog_dict, v, temp, mode='set')  # Set in new location
                if remove:
                    for k in remove:
                        CU.nested_dict_val(sweeplog_dict, k, mode='pop')
                if add:
                    for k, v in add.items():
                        CU.nested_dict_val(sweeplog_dict, k, v, mode='set')
            return sweeplog_dict
        else:
            raise NotFoundInHdfError(f'sweeplogs not found in "Experiment Copy/metadata/sweep_logs"')

    @with_hdf_read
    def get_default_data_infos(self) -> Dict[str, DataInfo]:
        """
        Get any default DataDescriptors to tell Data how to load experiment data

        Returns:
            (List[DataDescriptor]): list of Default data descriptors to use for loading Data
        """
        group = self.hdf.group.get('Default DataDescriptors')
        infos = {}
        for key in group:
            infos[key] = HDU.get_attr(group, key, dataclass=DataInfo)
        return infos

    def clear_caches(self):
        self.get_sweeplogs.cache_clear()

    @with_hdf_read
    def get_temperatures_dict(self):
        """Should return just the part of sweeplogs associated with Temperatures"""
        sweeplogs = self.get_sweeplogs()
        temperatures = sweeplogs.get('Temperature')
        return temperatures


