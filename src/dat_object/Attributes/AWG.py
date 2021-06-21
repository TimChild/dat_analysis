from __future__ import annotations
from src.dat_object.Attributes.DatAttribute import DatAttributeWithData
import numpy as np
from typing import Union, TYPE_CHECKING, Optional, Dict
import h5py
import logging
from src import hdf_util as HDU
from src import core_util as CU

if TYPE_CHECKING:
    from src.dat_object.Attributes.Logs import AWGtuple
    from src.dat_object.dat_hdf import DatHDF

logger = logging.getLogger(__name__)


class AWG(DatAttributeWithData):
    version = '2.0.0'
    group_name = 'AWG'
    description = 'FastDAC arbitrary wave generator information. Also provides some functionality for generating ' \
                  'mask waves etc.'

    def __init__(self, dat: DatHDF):
        self._info = None  # Used in some of the initializations so needs to be above super()
        super().__init__(dat)
        self._AWs = None
        self._freq = None

    def max(self, num: int = 0) -> float:
        """Returns max output of AW[num]"""
        return np.max(self.AWs[num][0])

    def min(self, num: int = 0) -> float:
        """Returns min output of AW[num]"""
        return np.min(self.AWs[num][0])

    @property
    def info(self) -> AWGtuple:
        if not self._info:
            self._info = self.get_group_attr('info')
        return self._info

    @property
    def AWs(self) -> Dict[int, np.ndarray]:
        if not self._AWs:
            self._AWs = {k: self.get_data(f'AW{k}') for k in self.info.outputs}
        return self._AWs

    @property
    def freq(self):
        if not self._freq:
            self._freq = self.measure_freq/self.info.wave_len
        return self._freq

    @property
    def measure_freq(self):
        return self.info.measureFreq

    @property
    def numpts(self):
        info = self.info
        return info.wave_len * info.num_cycles * info.num_steps

    @property
    def true_x_array(self):
        """The actual DAC steps of x_array"""
        x = self.get_data('x')
        return np.linspace(x[0], x[-1], self.info.num_steps)

    def get_single_wave(self, num: int) -> np.ndarray:
        """Returns a full single wave AW (with correct number of points for sample rate)"""
        if not self._check_wave_num(num):
            return None
        aw = self.AWs[num]
        return np.concatenate([np.ones(int(aw[1][i])) * aw[0][i] for i in range(aw.shape[1])])

    def get_full_wave(self, num) -> np.ndarray:
        """Returns the full waveform output through the whole scan with the same num points as x_array"""
        aw = self.get_single_wave(num)
        return np.array(list(aw) * int(self.info.num_cycles) * int(self.info.num_steps))

    def get_single_wave_masks(self, num: int) -> np.ndarray:
        """
        Returns single wave masks for arbitrary wave
        Args:
            num (int): Which AW

        Returns:
            List[np.ndarray, np.ndarray, np.ndarray]: A list of arrays of masks for AW
        """
        self._check_wave_num(num)
        aw = self.AWs[num]
        lens = aw[1].astype(int)
        masks = np.zeros((len(lens), np.sum(lens)), dtype=np.float16)  # Make 1 cycle
        for i, m in enumerate(masks):
            s = np.sum(lens[:i])
            m[s:s+lens[i]] = 1
            m[np.where(m == 0)] = np.nan
        return masks

    def get_full_wave_masks(self, num: int) -> np.ndarray:
        """
        Returns full wave masks for AW#
        Args:
            num (int): Which AW

        Returns:
            np.ndarray: An array of masks for AW (i.e. for 4 step wave, first dimension will be 4)
        """
        single_masks = self.get_single_wave_masks(num)
        full_masks = np.tile(single_masks, self.info.num_cycles * self.info.num_steps)
        return full_masks

    def eval(self, x, wave_num=0):
        """Returns square wave output at x value(s)

        Args:
            x (Union(int,float,np.ndarray)): x value(s) to get heating for
            wave_num (int): Which AWG to evaluate (0 or 1)
        Returns:
            (Union(float, np.ndarray)): Returns either the single value, or array of values
        """
        x = np.asanyarray(x)
        x_array = self.get_data('x')
        if x.shape == x_array.shape and np.all(np.isclose(x, x_array)):  # If full wave, don't bother searching for points
            idx = np.arange(x_array.shape[-1])
        else:
            idx = np.array(CU.get_data_index(x_array, x))
        wave = self.get_full_wave(wave_num)
        return wave[idx]

    def _check_wave_num(self, num):
        if num not in self.info.outputs.keys():
            raise HDU.NotFoundInHdfError(f'{num} not in AWs, choose from {self.info.outputs.keys()}')
        return True

    def initialize_minimum(self):
        self._copy_info_from_logs()
        self._copy_AW_descriptors()
        self.initialized = True

    def _copy_info_from_logs(self):
        awg_tuple: AWGtuple = self.dat.Logs.awg
        if awg_tuple is None:
            raise HDU.NotFoundInHdfError(f'AWG info not found in dat.Logs')
        self.set_group_attr('info', awg_tuple)

    def _copy_AW_descriptors(self):
        info = self.info   # Needs to be run AFTER copying info from Logs
        for k in info.outputs:
            descriptor = self.get_descriptor(f'fdAW_{k}')
            self.set_data_descriptor(descriptor, f'AW{k}')

    def clear_caches(self):
        self._info = None
        self._AWs = None
        self._freq = None



# class AWG(DatAttribute):
#     group_name = 'AWG'
#     version = '1.1'
#
#     """
#     Version changes:
#         1.1 -- Make general get_single_wave_mask, and full wave mask. No need for specific square one
#     """
#
#     def __init__(self, hdf):
#         super().__init__(hdf)
#         self.info: Optional[AWGtuple] = None
#         self.AWs: Optional[list] = None  # AWs as stored in HDF by exp (1 cycle with setpoints/samples)
#
#         # Useful for Square wave part to know some things about scan stored in other areas of HDF
#         self.x_array = None
#         self.measure_freq = None
#
#         self.get_from_HDF()
#
#     @property
#     def freq(self):
#         return self.measure_freq/self.info.wave_len
#
#     @property
#     def wave_duration(self):
#         """Length of once cycle of the wave in seconds"""
#         if self.measure_freq:
#             return self.info.wave_len*self.measure_freq
#         else:
#             logger.info(f'measure_freq not set for AWG')
#
#     @property
#     def true_x_array(self):
#         """The actual DAC steps of x_array"""
#         if self.x_array is not None:
#             return np.linspace(self.x_array[0], self.x_array[-1], self.info.num_steps)
#         else:
#             logger.info(f'x_array not set for AWG')
#
#     @property
#     def numpts(self):
#         info = self.info
#         return info.wave_len * info.num_cycles * info.num_steps
#
#     def _check_default_group_attrs(self):
#         super()._check_default_group_attrs()
#         self.group.attrs['description'] = "Information about Arbitrary Wave Generator used for scan. \n" \
#                                           "Also a place to store any AWG related data/results etc"
#
#     def get_from_HDF(self):
#         self.info = HDU.get_attr(self.group, 'Logs')  # Load NamedTuple in
#         if self.info is not None:
#             self.AWs = [self.group['AWs'].get(f'AW{k}') for k in self.info.outputs.keys()]
#         data_group = self.hdf.get('Data', None)
#         if data_group:
#             x_array = data_group.get('Exp_x_array', None)
#             if x_array:
#                 self.x_array = x_array
#         logs_group = self.hdf.get('Logs', None)
#         if logs_group:
#             fdac_group = logs_group.get('FastDACs', None)
#             if fdac_group:
#                 self.measure_freq = fdac_group.attrs.get('MeasureFreq', None)
#
#     def update_HDF(self):
#         logger.warning(f'Update HDF does not have any affect with AWG attribute currently')
#
#     def get_single_wave(self, num):
#         """Returns a full single wave AW (with correct number of points for sample rate)"""
#         if not self._check_wave_num(num): return None
#         aw = self.AWs[num]
#         return np.concatenate([np.ones(int(aw[1][i])) * aw[0][i] for i in range(aw.shape[1])])
#
#     def get_full_wave(self, num):
#         """Returns the full waveform output through the whole scan with the same num points as x_array"""
#         aw = self.get_single_wave(num)
#         return np.array(list(aw) * int(self.info.num_cycles) * int(self.info.num_steps))
#
#     def get_single_wave_masks(self, num):
#         """
#         Returns single wave masks for arbitrary wave
#         Args:
#             num (int): Which AW
#
#         Returns:
#             List[np.ndarray, np.ndarray, np.ndarray]: A list of arrays of masks for AW
#         """
#         self._check_wave_num(num, raise_error=True)
#         aw = self.AWs[num]
#         lens = aw[1].astype(int)
#         masks = np.zeros((len(lens), np.sum(lens)), dtype=np.float16)  # Make 1 cycle
#         for i, m in enumerate(masks):
#             s = np.sum(lens[:i])
#             m[s:s+lens[i]] = 1
#             m[np.where(m == 0)] = np.nan
#         return masks
#         # for sm in single_masks:
#         #     sm[np.where(sm == 0)] = np.nan
#         # return single_masks
#
#     def get_full_wave_masks(self, num):
#         """
#         Returns full wave masks for AW#
#         Args:
#             num (int): Which AW
#
#         Returns:
#             np.ndarray: An array of masks for AW (i.e. for 4 step wave, first dimension will be 4)
#         """
#         single_masks = self.get_single_wave_masks(num)
#         full_masks = np.tile(single_masks, self.info.num_cycles * self.info.num_steps)
#         return full_masks
#
#     def _check_wave_num(self, num, raise_error=False):
#         if num not in self.info.outputs.keys():
#             if raise_error is True:
#                 raise ValueError(f'{num} not in AWs, choose from {self.info.outputs.keys()}')
#             else:
#                 logger.warning(f'{num} not in AWs, choose from {self.info.outputs.keys()}')
#             return False
#         return True
#
#     def eval(self, x, wave_num=0):
#         """Returns square wave output at x value(s)
#
#         Args:
#             x (Union(int,float,np.ndarray)): x value(s) to get heating for
#             wave_num (int): Which AWG to evaluate (0 or 1)
#         Returns:
#             (Union(float, np.ndarray)): Returns either the single value, or array of values
#         """
#         x = np.asanyarray(x)
#         if x.shape == self.x_array.shape and np.all(np.isclose(x, self.x_array)):  # If full wave, don't bother searching for points
#             idx = np.arange(self.x_array.shape[-1])
#         else:
#             idx = np.array(CU.get_data_index(self.x_array, x))
#         wave = self.get_full_wave(wave_num)
#         return wave[idx]
#
#
# def init_AWG(group, logs_group, data_group: h5py.Group):
#     """Convert data from standardized experiment data to dat HDF
#     Should be run after Logs is initialized so that we can reuse AWG info saved there
#
#     Args:
#         AWG_tuple (AWGtuple): From Logs (contains sweeplogs info)
#         group (h5py.Group): AWG group in dat HDF
#         AWs (Union[List[np.ndarray], np.ndarray]): Either single, or list of AWs from Exp HDF
#
#     Returns:
#
#     """
#     wg = group.require_group('AWs')
#
#     AWG_tuple = HDU.get_attr(logs_group, 'AWG', None)
#     if AWG_tuple is None:
#         raise RuntimeError(f'No "AWG" found in Logs group, need to initialized there first')
#
#     # Get AWs from Exp_data and put in AWG/AWs
#     for k in AWG_tuple.outputs.keys():
#         wave = data_group.get(f'Exp_fdAW_{k}', None)
#         if wave is None:
#             logger.warning(f'fdAW_{k} was not found in HDF')
#         else:
#             wg[f'AW{k}'] = wave
#
#     # Add AWG logs info to AWG section directly by copying group
#     group.copy(logs_group['AWG'], group, 'Logs')
#
#     group.file.flush()
