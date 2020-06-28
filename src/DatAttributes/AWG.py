from src.DatAttributes.DatAttribute import DatAttribute
import numpy as np
from typing import Union
import h5py
from src.DatAttributes.Logs import AWGtuple
import logging
from src.HDF import Util as HDU
logger = logging.getLogger(__name__)


class AWG(DatAttribute):
    group_name = 'AWG'
    version = '1.0'

    def __init__(self, hdf):
        super().__init__(hdf)
        self.info: Union[AWGtuple, None] = None
        self.full_waves: Union[np.ndarray, None] = None  # put list of full AWG_waves here?

    def _set_default_group_attrs(self):
        super()._set_default_group_attrs()
        self.group.attrs['description'] = "Information about Arbitrary Wave Generator used for scan. \n" \
                                          "Also a place to store any AWG related data/results etc"

    def get_from_HDF(self):
        self.info = HDU.get_attr(self.group, 'Logs')  # Load NamedTuple in
        self.full_waves = np.array([self.group['AWs'].get(f'AW{k}') for k in self.info.output.keys()])

    def update_HDF(self):
        logger.warning(f'Update HDF does not have any affect with AWG attribute currently')


def init_AWG(group, logs_group, data_group: h5py.Group):
    """Convert data from standardized experiment data to dat HDF
    Should be run after Logs is initialized so that we can reuse AWG info saved there

    Args:
        AWG_tuple (AWGtuple): From Logs (contains sweeplogs info)
        group (h5py.Group): AWG group in dat HDF
        AWs (Union[List[np.ndarray], np.ndarray]): Either single, or list of AWs from Exp HDF

    Returns:

    """
    wg = group.require_group('AWs')

    AWG_tuple = HDU.get_attr(logs_group, 'AWG', None)
    if AWG_tuple is None:
        raise RuntimeError(f'No "AWG" found in Logs group, need to initialized there first')

    # Get AWs from Exp_data and put in AWG/AWs
    for k in AWG_tuple.outputs.keys():
        wave = data_group.get(f'Exp_fdAW_{k}', None)
        if wave is None:
            logger.warning(f'fdAW_{k} was not found in HDF')
        else:
            wg[f'AW{k}'] = wave

    # Add AWG logs info to AWG section directly by copying group
    group.copy(logs_group['AWG'], group, 'Logs')

    group.file.flush()
