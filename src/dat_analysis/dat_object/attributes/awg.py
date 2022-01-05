from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Dict
import logging

from ... import hdf_util as HDU
from ... import core_util as CU
from .dat_attribute import DatAttributeWithData

if TYPE_CHECKING:
    from .logs import AWGtuple
    from ..dat_hdf import DatHDF

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
        self._check_wave_num(num)
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


