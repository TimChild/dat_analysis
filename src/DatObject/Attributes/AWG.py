from src.DatObject.Attributes.DatAttribute import DatAttribute
import numpy as np
from typing import Union, TYPE_CHECKING
import h5py
import logging
from src import HDF_Util as HDU

if TYPE_CHECKING:
    from src.DatObject.Attributes.Logs import AWGtuple

logger = logging.getLogger(__name__)


class SquareWaveMixin(object):
    info = None  # type: AWGtuple
    get_single_wave = None

    def get_step_ids(self, num):
        """Returns index positions of steps (i.e. when DAC should be at a new value)"""
        awg = self.info
        aw = self.get_single_wave(num)
        samples = aw[1]
        steps = [samples]


class AWG(SquareWaveMixin, DatAttribute):
    group_name = 'AWG'
    version = '1.0'

    def __init__(self, hdf):
        super().__init__(hdf)
        self.info: Union[AWGtuple, None] = None
        self.AWs: Union[np.ndarray, None] = None  # AWs as stored in HDF by exp (1 cycle with setpoints/samples)

    def _set_default_group_attrs(self):
        super()._set_default_group_attrs()
        self.group.attrs['description'] = "Information about Arbitrary Wave Generator used for scan. \n" \
                                          "Also a place to store any AWG related data/results etc"

    def get_from_HDF(self):
        self.info = HDU.get_attr(self.group, 'Logs')  # Load NamedTuple in
        self.AWs = np.array([self.group['AWs'].get(f'AW{k}') for k in self.info.outputs.keys()])

    def update_HDF(self):
        logger.warning(f'Update HDF does not have any affect with AWG attribute currently')

    def get_single_wave(self, num):
        """Returns a full single wave AW (with correct number of points for sample rate)"""
        if num not in self.info.outputs.keys():
            logger.warning(f'{num} not in AWs, choose from {self.info.outputs.keys()}')
            return None
        aw = self.AWs[num]
        return np.concatenate([np.ones(int(aw[1, i])) * aw[0, i] for i in range(aw.shape[1])])

    def get_full_wave(self, num):
        """Returns the full waveform output through the whole scan with the same num points as x_array"""
        aw = self.get_single_wave(num)
        return np.array(list(aw)*int(self.info.num_cycles)*int(self.info.num_steps))





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
