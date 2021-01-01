from __future__ import annotations
from src.HDF_Util import with_hdf_write, with_hdf_read
from typing import TYPE_CHECKING, Optional, Dict, List
import copy
import numpy as np
from scipy.interpolate import interp1d
from .DatAttribute import DatAttributeWithData, DatDataclassTemplate, FitInfo
import src.CoreUtil as CU

if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF
    from src.DatObject.Attributes import AWG

from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# SETTLE_TIME = 1.2e-3  # 9/8/20 -- measured to be ~0.8ms, so using 1.2ms to be safe
SETTLE_TIME = 5e-3  # 9/10/20 -- measured to be ~3.75ms, so using 5ms to be safe (this is with RC low pass filters)


class SquareEntropy(DatAttributeWithData):
    version = '2.0.0'
    group_name = 'Square Entropy'
    description = 'Working with square entropy stuff to generate entropy signal for dat.Entropy (which looks for ' \
                  'dat.SquareEntropy.get_entropy_signal()'

    def __init__(self, dat: DatHDF):
        super().__init__(dat)
        self._Outputs: Dict[str, Output] = {}
        self._square_awg = None

    @property
    def square_awg(self) -> AWG:
        """
        Copy of the normal AWG but where AWs are forced to be 4 points only (i.e. groups in possible ramps into the main 4 parts)
        Assumes equal length to each part

        Note: square_awg still has access to HDF. Tried to this from overwriting things, but maybe it's still possible so be careful.
        Returns:
        """
        if not self._square_awg:
            awg = copy.copy(self.dat.AWG)

            # Try to prevent this overwriting real AWG info in HDF by removing set methods
            awg.set_group_attr = None
            awg.set_data = None
            awg.set_data_descriptor = None

            awg._AWs = {k: _force_four_point_AW(aw) for k, aw in awg.AWs.items()}
            self._square_awg = awg
        return self._square_awg

    @property
    def default_Input(self) -> Input:
        return self.get_Inputs()

    @property
    def default_ProcessParams(self) -> ProcessParams:
        return self.get_ProcessParams()

    @property
    def default_Output(self) -> Output:
        return self.get_Outputs()

    @property
    def x(self):
        """Default x array for Square Entropy (i.e. x per DAC step)"""
        return self.default_Output.x

    @property
    def entropy_signal(self) -> np.ndarray:
        """Default entropy signal for Square Entropy"""
        return self.default_Output.entropy_signal

    @property
    def avg_entropy_signal(self) -> np.ndarray:
        """Default averaged entropy signal for Square Entropy"""
        return self.default_Output.average_entropy_signal

    def get_Inputs(self,
                   name: Optional[str] = None,
                   x_array: Optional[np.ndarray] = None,
                   i_sense: Optional[np.ndarray] = None,
                   num_steps: Optional[int] = None,
                   num_cycles: Optional[int] = None,
                   setpoint_lengths: Optional[List[int]] = None,
                   full_wave_masks: Optional[np.ndarray] = None,
                   centers: Optional[np.ndarray] = None,
                   avg_nans: Optional[bool] = None,
                   save_name: Optional[str] = None) -> Input:
        """
        Gathers together necessary inputs for processing square wave info. Anything not specified is gathered by
        defaults.
        If a name is specified and that entry has been saved in the HDF, it will use that as a starting point and
        change anything else that is specified.
        Use save_name to save the Input to HDF, otherwise it WILL NOT be stored automatically

        Returns:
        Args:
            name (): Look for stored Inputus with this name
            x_array (): Original x_array
            i_sense (): Charge sensor data (1D or 2D)
            num_steps (): Number of DAC steps
            num_cycles (): Number of full square wave cycles per DAC step
            setpoint_lengths (): Number of readings per setpoint of Square Wave
            full_wave_masks (): Mask for chunking data into setpoints
            centers (): Center positions to use for averaging data (if left as None then v0 parts of i_sense data will
                be fit with i_sense and that will be used. Good but slow). !! Saved in Outputs.centers_used. !!
            avg_nans (): Whether to include columns which contain NaNs in averaging (generally safer not to but a single
                bad row will mean that almost everything is thrown out)
            save_name (): Name to save into HDF with (i.e. to be accessed with name later)

        Returns:
            (Inputs): The Inputs which go into processing square wave info
        """

        inp: Optional[Input] = None
        if name:
            inp = self._get_saved_Inputs(name)
            if inp:
                for k, v in {'x_array': x_array, 'i_sense': i_sense, 'num_steps': num_steps, 'num_cycles': num_cycles,
                             'setpoint_lengths': setpoint_lengths, 'full_wave_masks': full_wave_masks, 'centers': centers,
                             'avg_nans': avg_nans}.items():
                    if v is not None:  # Only overwrite things that have changed
                        setattr(inp, k, v)

        if not inp:  # If not found above then create a new inp
            if any([v is None for v in [num_steps, num_cycles, setpoint_lengths, full_wave_masks]]):
                awg = self.square_awg
                if not num_steps:
                    num_steps = awg.info.num_steps
                if not num_cycles:
                    num_cycles = awg.info.num_cycles
                if not setpoint_lengths:
                    setpoint_lengths = _force_four_point_AW(awg.AWs[0])[1]  # Assume first AW works here
                if not full_wave_masks:
                    full_wave_masks = awg.get_full_wave_masks(0)  # Assume first AW works here

            if x_array is None:
                x_array = self.get_data('x')
            if i_sense is None:
                i_sense = self.get_data('i_sense')
            if centers is None:
                pass  # If pass in None, then centers are calculated from v0 parts of i_sense
            if avg_nans is None:
                avg_nans = False  # Safer to throw away columns with nans in when averaging in general
            inp = Input(x_array=x_array, i_sense=i_sense, num_steps=num_steps, num_cycles=num_cycles,
                        setpoint_lengths=setpoint_lengths, full_wave_masks=full_wave_masks, centers=centers,
                        avg_nans=avg_nans)

        if save_name:
            self.set_group_attr(save_name, inp, group_name='/'.join([self.group_name, 'Inputs']), DataClass=Input)
        return inp

    def _get_saved_Inputs(self, name):
        inp = self.get_group_attr(name, check_exists=True, group_name='/'.join([self.group_name, 'Inputs']), DataClass=Input)
        return inp

    def get_ProcessParams(self,
                          name: Optional[str] = None,
                          setpoint_start: Optional[int] = None,
                          setpoint_fin: Optional[int] = None,
                          cycle_start: Optional[int] = None,
                          cycle_fin: Optional[int] = None,
                          save_name: Optional[str] = None,
                          ) -> ProcessParams:
        """
        Gathers together neccessary ProcessParams info. Similar to get_Inputs.
        If a name is specified and has been saved in HDF, that will be used as a starting point, and anything else
        specified will be changed (NOTE: passing in None will not overwrite things, use 0 or e.g. len(setpoint) to
        refer to beginning or end of array in that case).
        Use save_name to save the ProcessParams to HDF, otherwise they WILL NOT be stored automatically

        Args:
            name ():  Look for stored ProcessParams with this name
            setpoint_start (): Where to start averaging data each setpoint
            setpoint_fin (): Where to finish averaging data each setpoint
            cycle_start (): Where to start averaging cycles each DAC step
            cycle_fin (): Where to finish averaging cycles each DAC step
            save_name (): Name to save under in HDF

        Returns:
            (ProcessParams): Filled ProcessParams

        """
        pp: Optional[ProcessParams] = None
        if name:
            pp = self._get_saved_ProcessParams(name)

        if not pp:
            # None defaults are what I want here anyway
            pp = ProcessParams(setpoint_start=setpoint_start, setpoint_fin=setpoint_fin, cycle_start=cycle_start,
                               cycle_fin=cycle_fin)
        else:
            if setpoint_start:
                pp.setpoint_start = setpoint_start
            if setpoint_fin:
                pp.setpoint_fin = setpoint_fin
            if cycle_start:
                pp.cycle_start = cycle_start
            if cycle_fin:
                pp.cycle_fin = cycle_fin

        if save_name:
            self.set_group_attr(save_name, pp, group_name='/'.join([self.group_name, 'ProcessParams']), DataClass=ProcessParams)
        return pp

    def _get_saved_ProcessParams(self, name: str):
        pp = self.get_group_attr(name, check_exists=True, group_name='/'.join([self.group_name, 'ProcessParams']), DataClass=ProcessParams)
        return pp

    def get_Outputs(self, name: Optional[str] = 'default', inputs: Optional[Input] = None,
                    process_params: Optional[ProcessParams] = None, overwrite=False) -> Output:
        """
        Either looks for saved Outputs in HDF file, or generates new Outputs given Inputs and/or ProcessParams.

        If <name> AND Inputs, ProcessParams passed in and <name> already exists, WILL NOT overwrite unless 'overwrite'
        is set True. Otherwise Outputs will be saved under <name>

        Args:
            name (): Name to look for / save under
            inputs (): Input data for calculating Outputs
            process_params (): ProcessParams for calculating Outputs
            overwrite (bool): If False, previously calculated is returned if exists, otherwise overwritten

        Returns:
            (Outputs): All the various data after processing

        """
        if name and not overwrite:
            if name in self._Output_names():
                out = self._get_saved_Outputs(name)
                return out  # No need to go further if found

        if not inputs:
            inputs = self.get_Inputs()
        if not process_params:
            process_params = self.get_ProcessParams()

        out = process(inputs, process_params)
        self._save_Outputs(name, out)
        return out

    @with_hdf_read
    def _Output_names(self):
        """Get names of saved Outputs in HDF"""
        group = self.hdf.group.get('Outputs')
        return list(group.keys())  # Assume everything in Outputs is an Output


    def _get_saved_Outputs(self, name):
        gn = '/'.join([self.group_name, 'Outputs'])
        if not name in self._Outputs:
            self._Outputs[name] = self.get_group_attr(name, check_exists=True, group_name=gn, DataClass=Output)
        return self._Outputs[name]

    def _save_Outputs(self, name: str, out: Output):
        gn = '/'.join([self.group_name, 'Outputs'])
        self._Outputs[name] = out
        self.set_group_attr(name, out, group_name=gn, DataClass=Output)  # Always should be saving if getting to here

    def initialize_minimum(self):
        self._set_default_data_descriptors()
        self._make_groups()
        self.initialized = True

    def _set_default_data_descriptors(self):
        data_keys = ['x', 'i_sense']
        for key in data_keys:
            descriptor = self.get_descriptor(key)
            self.set_data_descriptor(descriptor, key)

    @with_hdf_write
    def _make_groups(self):
        self.hdf.group.require_group('Inputs')
        self.hdf.group.require_group('ProcessParams')
        self.hdf.group.require_group('Outputs')

    def clear_caches(self):
        self._Outputs = {}
        self._square_awg = None


@dataclass
class Input(DatDataclassTemplate):
    x_array: np.ndarray
    i_sense: np.ndarray
    num_steps: int
    num_cycles: int
    setpoint_lengths: List[int]
    full_wave_masks: np.ndarray
    centers: Optional[np.ndarray] = None
    avg_nans: bool = False


@dataclass
class ProcessParams(DatDataclassTemplate):
    setpoint_start: Optional[int]  # Index to start averaging for each setpoint
    setpoint_fin: Optional[int]  # Index to stop averaging for each setpoint
    cycle_start: Optional[int]  # Index to start averaging cycles
    cycle_fin: Optional[int]  # Index to stop averaging cycles


@dataclass
class Output(DatDataclassTemplate):
    # Data that will be calculated
    x: np.ndarray = field(default=None, repr=False)  # x_array with length of num_steps (for cycled, averaged, entropy)
    chunked: np.ndarray = field(default=None, repr=False)  # Data broken in to chunks based on AWG (just plot
    # raw_data on orig_x_array)
    setpoint_averaged: np.ndarray = field(default=None, repr=False)  # Setpoints averaged only
    setpoint_averaged_x: np.ndarray = field(default=None, repr=False)  # x_array for setpoints averaged only
    cycled: np.ndarray = field(default=None, repr=False)  # setpoint averaged and then cycles averaged data
    averaged: np.ndarray = field(default=None, repr=False)  # setpoint averaged, cycle_avg, then averaged in y

    centers_used: np.ndarray = None
    entropy_signal: np.ndarray = field(default=None, repr=False)  # 2D Entropy signal data
    average_entropy_signal: np.ndarray = field(default=None, repr=False)  # Averaged Entropy signal


def process(input_info: Input, process_pars: ProcessParams) -> Output:
    output = Output()
    inp = input_info
    pp = process_pars

    # Calculate true x_array (num_steps)
    output.x = np.linspace(inp.x_array[0], inp.x_array[-1], inp.num_steps)

    # Get chunked data (setpoints, ylen, numsteps, numcycles, splen)
    output.chunked = chunk_data(inp.i_sense, full_wave_masks=inp.full_wave_masks, setpoint_lengths=inp.setpoint_lengths,
                                num_steps=inp.num_steps, num_cycles=inp.num_cycles)

    # Average setpoints of data ([ylen], setpoints, numsteps, numcycles)
    output.setpoint_averaged = average_setpoints(output.chunked, start_index=pp.setpoint_start,
                                                 fin_index=pp.setpoint_fin)
    output.setpoint_averaged_x = np.linspace(inp.x_array[0], inp.x_array[-1],
                                             inp.num_steps * inp.num_cycles)

    # Averaged cycles ([ylen], setpoints, numsteps)
    output.cycled = average_cycles(output.setpoint_averaged, start_cycle=pp.cycle_start, fin_cycle=pp.cycle_fin)

    # Center and average 2D data or skip for 1D
    output.x, output.averaged, output.centers_used = average_2D(output.x, output.cycled, centers=inp.centers, avg_nans=inp.avg_nans)

    # region Use this if want to start shifting each heater setpoint of data left or right
    # Align data
    # output.x, output.averaged = align_setpoint_data(xs, output.averaged, nx=None)
    # endregion

    # Entropy signal
    output.entropy_signal = entropy_signal(np.moveaxis(output.cycled, 1, 0))  # Moving setpoint axis to be first
    output.average_entropy_signal = entropy_signal(output.averaged)

    return output


def _force_four_point_AW(aw: np.ndarray):
    """
    Takes an single AW and returns an AW with 4 setpoints
    Args:
        aw (np.ndarray):

    Returns:
        np.ndarray: AW with only 4 setpoints but same length as original
    """
    aw = np.asanyarray(aw)
    assert aw.ndim == 2
    full_len = np.sum(aw[1])
    assert full_len % 4 == 0
    new_aw = np.ndarray((2, 4), np.float32)

    # split Setpoints/lens into 4 sections
    for i, aw_chunk in enumerate(np.reshape(aw.swapaxes(0, 1), (4, -1, 2)).swapaxes(1, 2)):
        sp = aw_chunk[0, -1]  # Last value of chunk (assuming each setpoint handles it's own ramp)
        length = np.sum(aw_chunk[1])
        new_aw[0, i] = sp
        new_aw[1, i] = length
    return new_aw


# region Processing functions from I_sense to Entropy
"""All the functions for processing I_sense data into the various steps of square wave heated data"""


def chunk_data(data, full_wave_masks: np.ndarray, setpoint_lengths: List[int], num_steps: int, num_cycles: int) -> List[np.ndarray]:
    """
    Breaks up data into chunks which make more sense for square wave heating datasets.
    Args:
        data (np.ndarray): 1D or 2D data (full data to match original x_array).
            Note: will return with y dim regardless of 1D or 2D

    Returns:
        List[np.ndarray]: Data broken up into chunks [setpoints, np.ndarray(ylen, num_steps, num_cycles, sp_len)].

            NOTE: Has to be a list returned and not a ndarray because sp_len may vary per steps

            NOTE: This is the only step where setpoints should come first, once sp_len binned it should be ylen first
    """
    masks = full_wave_masks
    zs = []
    for mask, sp_len in zip(masks, setpoint_lengths):
        sp_len = int(sp_len)
        z = np.atleast_2d(data)  # Always assume 2D data
        zm = z * mask  # Mask data
        zm = zm[~np.isnan(zm)]  # remove blanks
        zm = zm.reshape(z.shape[0], num_steps, num_cycles, sp_len)
        zs.append(zm)
    return zs


def average_setpoints(chunked_data, start_index=None, fin_index=None):
    """ Averages last index of AWG data passed in from index s to f.

    Args:
        chunked_data (List[np.ndarray]): List of datas chunked nicely for AWG data.
            dimensions (num_setpoints_per_cycle, (len(y), num_steps, num_cycles, sp_len))
        start_index (Union(int, None)): Start index to average in each setpoint chunk
        fin_index (Union(int, None)): Final index to average to in each setpoint chunk (can be negative)

    Returns:
        np.ndarray: Array of zs with averaged last dimension. ([ylen], setpoints, num_steps, num_cycles)
        Can be an array here because will always have 1 value per
        averaged chunk of data (i.e. can't have different last dimension any more)
    """

    assert np.all([arr.ndim == 4 for arr in chunked_data])  # Assumes [setpoints, (ylen, num_steps, num_cycles, sp_len)]
    nz = []
    for z in chunked_data:
        z = np.moveaxis(z, -1, 0)  # move sp_len to first axis to make mean nicer
        nz.append(np.mean(z[start_index:fin_index], axis=0))

    # nz = [np.mean(z[:, :, :, start_index:fin_index], axis=3) for z in chunked_data]  # Average the last dimension
    nz = np.moveaxis(np.array(nz), 0, 1)  # So that ylen is first now
    # (ylen, setpoins, num_steps, num_cycles)

    if nz.shape[0] == 1:  # Remove ylen dimension if len == 1
        nz = np.squeeze(nz, axis=0)
    return np.array(nz)


def average_cycles(binned_data, start_cycle=None, fin_cycle=None):
    """
    Average values from cycles from start_cycle to fin_cycle
    Args:
        binned_data (np.ndarray): Binned AWG data with shape ([ylen], setpoints, num_steps, num_cycles)
        start_cycle (Union(int, None)): Cycle to start averaging from
        fin_cycle (Union(int, None)): Cycle to finish averaging on (can be negative to count backwards)

    Returns:
        np.ndarray: Averaged data with shape ([ylen], setpoints, num_steps)

    """
    # [y], setpoints, numsteps, cycles
    data = np.array(binned_data, ndmin=4)  # [y], setpoints, numsteps, cycles
    averaged = np.mean(np.moveaxis(data, -1, 0)[start_cycle:fin_cycle], axis=0)
    if averaged.shape[0] == 1:  # Return 1D or 2D depending on y_len
        averaged = np.squeeze(averaged, axis=0)
    return averaged


def average_2D(x: np.ndarray, data: np.ndarray, centers: Optional[np.ndarray] = None, avg_nans: bool = False):
    """
    Averages data in y direction after centering using fits to v0 parts of square wave. Returns 1D data unchanged
    Args:
        x (np.ndarray): Original x_array for data
        data (np.ndarray): Data after binning and cycle averaging. Shape ([ylen], setpoints, num_steps)
        centers (Optional[np.ndarray]): Optional center positions to use instead of standard automatic transition fits
        avg_nans (bool): Whether to average data which includes NaNs (useful for two part entropy scans)
    Returns:
        Tuple[np.ndarray, np.ndarray]: New x_array, averaged_data (shape (setpoints, num_steps))
    """
    if data.ndim == 3:
        z0s = data[:, (0, 2)]
        z0_avg_per_row = np.mean(z0s, axis=1)
        if centers is None:
            from .Transition import transition_fits
            fits = transition_fits(x, z0_avg_per_row)
            if np.any([fit is None for fit in fits]):  # Not able to do transition fits for some reason
                logger.warning(f'{np.sum([1 if fit is None else 0 for fit in fits])} transition fits failed, blind '
                               f'averaging instead of centered averaging')
                return x, np.mean(data, axis=0)
            fit_infos = [FitInfo.from_fit(fit) for fit in fits]  # Has my functions associated
            centers = [fi.best_values.mid for fi in fit_infos]
        nzs = []
        nxs = []
        for z in np.moveaxis(data, 1, 0):  # For each of v0_0, vP, v0_1, vM
            nz, nx = CU.center_data(x, z, centers, return_x=True)
            nzs.append(nz)
            nxs.append(nx)
        assert (nxs[0] == nxs).all()  # Should all have the same x_array
        ndata = np.array(nzs)
        if avg_nans is True:
            ndata = np.nanmean(ndata, axis=1)  # Average centered data
        else:
            ndata = np.mean(ndata, axis=1)  # Average centered data
        nx = nxs[0]
    else:
        nx = x
        ndata = data
        logger.info(f'Data passed in was likely 1D already, same values returned')
    return nx, ndata, centers


def entropy_signal(data):
    """
    Calculates equivalent of second harmonic from data with v0_0, vP, v0_1, vM as first dimension
    Note: Data should be aligned for same x_array before doing this
    Args:
        data (np.ndarray): Data with first dimension corresponding to v0_0, vP, v0_1, vM. Can be any dimensions for rest

    Returns:
        np.ndarray: Entropy signal array with same shape as data minus the first axis

    """
    assert data.shape[0] == 4
    entropy_data = -1 * (np.mean(data[(1, 3),], axis=0) - np.mean(data[(0, 2),], axis=0))
    return entropy_data


def integrate_entropy(data, sf):
    return np.nancumsum(data) * sf


def calculate_dT(bias_lookup, bias):
    return bias_lookup[bias] - bias_lookup[0]


def align_setpoints(xs, data, nx=None):
    """
    In case want to realign data where each setpoint of heating has a different x_array (i.e. taking into account some
    additional shifts)
    Args:
        xs (np.ndarray):  x_array for each heating setpoint in data
        data (np.ndarray):  data with shape (setpoints, num_steps)
        nx (np.ndarray): New x_array to put data on, or will use first of xs by default

    Returns:
        Tuple[np.ndarray, np.ndarray]: new x_array, interpolated data with same shape as original
    """
    assert xs.ndim == 2  # different x_array for each heating setpoint
    assert xs.shape[0] == data.shape[0]
    oxs = xs  # Old xs
    if nx is None:
        nx = xs[0]  # New x
    ndata = []  # New data
    for ox, z in zip(oxs, data):
        interper = interp1d(xs, z, bounds_error=False)
        ndata.append(interper(nx))
    data = np.array(ndata)  # Data which shares the same x axis
    return nx, data


# endregion


#
#
# class OldSquareEntropy(DA.DatAttribute):
#     version = '1.0'
#     group_name = 'SquareEntropy'
#
#     def __init__(self, hdf):
#         super().__init__(hdf)
#         self.x = None
#         self.y = None
#         self.data = None
#         self.Processed: Optional[SquareProcessed] = None
#         self.SquareAWG: Optional[SquareWaveAWG] = None
#         self.get_from_HDF()
#
#         # For temp storage
#         self._entropy_data = None
#
#     @property
#     def entropy_data(self):
#         if self._entropy_data is None and CU.get_nested_attr_default(self, 'Processed.outputs.cycled', None) is not None:
#             data = CU.get_nested_attr_default(self, 'Processed.outputs.cycled', None)
#             self._entropy_data = entropy_signal(np.swapaxes(data, 0, 1))  # Put 4 parts as first axis, then returns 2D
#         return self._entropy_data
#
#     @property
#     def dS(self):
#         return self.Processed.outputs.entropy_fit.best_values.dS
#
#     @property
#     def ShowPlots(self):
#         return self.Processed.plot_info.show
#
#     @ShowPlots.setter
#     def ShowPlots(self, value):
#         assert isinstance(value, ShowPlots)
#         self.Processed.plot_info.show = value
#
#     def get_from_HDF(self):
#         super().get_from_HDF()  # Doesn't do much
#         dg = self.group.get('Data', None)
#         if dg is not None:
#             self.x = dg.get('x', None)
#             self.y = dg.get('y', None)
#             if isinstance(self.y, float) and np.isnan(self.y):  # Because I store None as np.nan
#                 self.y = None
#             self.data = dg.get('i_sense', None)
#         self.Processed = self._get_square_processed()
#         self.ShowPlots = self.Processed.plot_info.show
#         self.SquareAWG = SquareWaveAWG(self.hdf)
#
#     def _get_square_processed(self):
#         awg = SquareWaveAWG(self.hdf)
#         inp = Input(self.data, self.x, awg, bias=None, transition_amplitude=None)
#         spg = self.group.get('SquareProcessed')
#         if spg is not None:
#             sp_data = dict()
#             sp_data['Input'] = inp
#             for name, dc, sdc in zip(['ProcessParams', 'Outputs', 'PlotInfo'], [ProcessParams, Output, PlotInfo], [{}, {'integrated_info': IntegratedInfo}, {'show': ShowPlots}]):
#                 g = spg.get(name)
#                 if g is not None:
#                     sp_data[name] = dataclass_from_group(g, dc=dc, sub_dataclass=sdc)
#             # Make dict with correct keys for SquareProcessed
#             spdata = {k: sp_data.pop(o_k) for k, o_k in zip(['inputs', 'process_params', 'outputs', 'plot_info'],
#                                                             ['Input', 'ProcessParams', 'Outputs', 'PlotInfo'])}
#             sp = SquareProcessed(**spdata)
#         else:
#             sp = SquareProcessed(process_params=ProcessParams(setpoint_start=int(np.round(SETTLE_TIME*awg.measure_freq))))
#         return sp
#
#     def update_HDF(self):
#         super().update_HDF()
#         # self.group.attrs['description'] =
#         dg = self.group.require_group('Data')
#         for name, data in zip(['x', 'y', 'i_sense'], [self.x, self.y, self.data]):
#             if data is None:
#                 data = np.nan
#             HDU.set_data(dg, name, data)  # Removes dataset before setting if necessary
#         self._set_square_processed()
#         self.hdf.flush()
#
#     def _set_square_processed(self):
#         sp = self.Processed
#         spg = self.group.require_group('SquareProcessed')
#         # inpg = spg.require_group('Inputs')  # Only drawn from HDF data anyway
#         ppg = spg.require_group('ProcessParams')
#         outg = spg.require_group('Outputs')
#         pig = spg.require_group('PlotInfo')
#         # dataclass_to_group(inpg, sp.inputs)
#         dataclass_to_group(ppg, sp.process_params)
#         dataclass_to_group(outg, sp.outputs)
#         dataclass_to_group(pig, sp.plot_info, ignore=['axs', 'axs_dict'])
#
#     def _check_default_group_attrs(self):
#         super()._check_default_group_attrs()
#
#     def process(self):
#         awg = SquareWaveAWG(self.hdf)
#         # transition = T.NewTransitions(self.hdf)
#         assert awg is not None
#         # assert transition is not None
#         sp = self.Processed if self.Processed else SquareProcessed()
#
#         inp = sp.inputs
#         if None in [inp.raw_data, inp.orig_x_array, inp.awg]: # Re init if necessary
#             inp = Input(raw_data=self.data, orig_x_array=self.x, awg=awg, bias=None, transition_amplitude=None)
#
#         # Use already stored process_params (which default to reasonable settings anyway)
#         pp = sp.process_params
#
#         # Recalculate ouptuts
#         out = process(inp, pp)
#
#         # Keep same plot_info as previous (or default)
#         plot_info = sp.plot_info
#
#         sp = SquareProcessed(inp, pp, out, plot_info)
#         self.Processed = sp
#         self.update_HDF()

#
# def dataclass_to_group(group: h5py.Group, dc, ignore=None):
#     """
#     Stores all values from dataclass into group, can be used to re init the given dataclass later
#     Args:
#         group (h5py.Group):
#         dc (dataclass):
#         ignore (Union(List[str], str)): Any Dataclass entries not to store (definitely anything that is not in init!!)
#     Returns:
#         (None):
#     """
#     ignore = ignore if ignore else list()
#     dc_path = '.'.join((dc.__class__.__module__, dc.__class__.__name__))
#     HDU.set_attr(group, 'Dataclass', dc_path)
#     group.attrs['description'] = 'Dataclass'
#     assert is_dataclass(dc)
#     # for k, v in asdict(dc).items():  # Tries to be too clever about inner attributes (e.g. lm.Parameters doesn't work)
#     for k in dc.__annotations__:
#         v = getattr(dc, k)
#         if k not in ignore:
#             if isinstance(v, (np.ndarray, h5py.Dataset)):
#                 HDU.set_data(group, k, v)
#             elif isinstance(v, list):
#                 HDU.set_list(group, k, v)
#             elif isinstance(v, DA.FitInfo):
#                 v.save_to_hdf(group, k)
#             elif is_dataclass(sub_dc := getattr(dc, k)):
#                 sub_group = group.require_group(k)
#                 dataclass_to_group(sub_group, sub_dc)
#             else:
#                 HDU.set_attr(group, k, v)
#
#
# def dataclass_from_group(group, dc, sub_dataclass=None):
#     """
#     Restores dataclass from HDF
#     Args:
#        group (h5py.Group):
#         dataclass (dataclass):
#         sub_dataclass (Optional[dict]): Dict of key: Dataclass for any sub_dataclasses
#
#     Returns:
#         (dataclass): Returns filled dataclass instance
#     """
#     assert group.attrs.get('description') == 'Dataclass'
#
#     all_keys = set(group.keys()).union(set(group.attrs.keys())) - {'Dataclass', 'description'}
#
#     d = dict()
#     for k in all_keys:
#         v = HDU.get_attr(group, k)
#         if v is None:  # For loading datasets, and if it really doesn't exist, None will be returned again
#             v = group.get(k, None)
#             if isinstance(v, h5py.Group):
#                 description = v.attrs.get('description')
#                 if description == 'list':
#                     v = HDU.get_list(group, k)
#                 elif description == 'FitInfo':
#                     v = DA.fit_group_to_FitInfo(v)
#                 elif description == 'Dataclass':
#                     if k in sub_dataclass:
#                         v = dataclass_from_group(v, sub_dataclass[k])
#                     else:
#                         raise KeyError(f'Trying to load a dataclass [{k}] without specifying the Dataclass type. Please provide in sub_dataclass')
#                 elif description is None and not v.keys() and not v.attrs.keys():  # Empty Group... Nothing saved in it
#                     v = None
#                 else:
#                     logger.warning(f'{v} is unexpected entry which seems to contain some data. None returned')
#                     v = None
#             elif isinstance(v, h5py.Dataset):
#                 v = v[:]
#         d[k] = v
#     initialized_dc: dataclass = dc(**d)
#     return initialized_dc
#

# region Modelling Only
""" Override SquareWaveAWG class to allow it to be created as a model
Also includes modelling function in this section"""

# @dataclass(init=False)  # Using to make nice repr etc, but the values will be init from HDF
# class SquareWaveAWG(AWG.AWG):
#     v0: float
#     vp: float
#     vm: float
#
#     def __init__(self, hdf):
#         super().__init__(hdf)
#
#
#     def get_from_HDF(self):
#         super().get_from_HDF()
#         self.ensure_four_setpoints()  # Even if ramps in wave, this will make it look like a 4 point square wave
#         square_aw = self.AWs[0]  # Assume AW0 for square wave heating
#         if square_aw.shape[-1] == 4:
#             self.v0, self.vp, _, self.vm = square_aw[0]
#         else:
#             logger.warning(f'Unexpected shape of square wave output: {square_aw.shape}')
#
#     def ensure_four_setpoints(self):
#         """
#         This turns a arbitrary looking wave into 4 main setpoints assuming 4 equal lengths and that the last value in
#         each section is the true setpoint.
#
#         This should fix self.get_full_wave_masks() as well
#
#         Assuming that square wave heating is always done with 4 setpoints (even if AWs contain more). This is to allow
#         for additional ramp behaviour between setpoints. I.e. 4 main setpoints but with many setpoints to define
#         ramps in between.
#         """
#         if self.AWs is not None:
#             new_AWs = list()
#             for aw in self.AWs:
#                 new_AWs.append(_force_four_point_AW(aw))  # Turns any length AW into a 4 point Square AW
#             self.AWs = np.array(new_AWs)
#
#
# @dataclass
# class SquareAWGModel(SquareWaveAWG):
#     measure_freq: float = 1000
#     start: InitVar[float] = -10
#     fin: InitVar[float] = 10
#     sweeprate: InitVar[float] = 1
#
#     v0: float = 0
#     vp: float = 100
#     vm: float = -100
#
#     step_duration: InitVar[float] = 0.25
#     _step_dur: Union[float, None] = field(default=None, repr=False, init=False)
#
#     def __post_init__(self, start: float, fin: float, sweeprate: float, step_duration: float):
#         self.step_dur = step_duration
#         self.num_steps = None  # Because self.info is called in get_numsteps and includes num_steps
#         self.num_steps = self._get_numsteps(sweeprate, start, fin)
#         self.x_array = np.linspace(start, fin, self.numpts)
#
#     @property
#     def step_dur(self):
#         return self._step_dur
#
#     @step_dur.setter
#     def step_dur(self, value):
#         self._step_dur = round(value * self.measure_freq) / self.measure_freq
#
#     @property
#     def info(self):
#         wave_len = self.step_dur * self.measure_freq * 4
#         assert np.isclose(round(wave_len), wave_len, atol=0.00001)  # Should be an int
#         return Logs.AWGtuple(outputs={0: [0]},  # wave 0 output 0
#                              wave_len=int(wave_len),
#                              num_adcs=1,
#                              samplingFreq=self.measure_freq,
#                              measureFreq=self.measure_freq,
#                              num_cycles=1,
#                              num_steps=self.num_steps)
#
#     @property
#     def AWs(self):
#         step_samples = round(self.step_dur * self.measure_freq)
#         assert np.isclose(round(step_samples), step_samples, atol=0.00001)  # Should be an int
#         return [np.array([[self.v0, self.vp, self.v0, self.vm],
#                           [int(step_samples)] * 4])]
#
#     def _get_numsteps(self, sweeprate, start, fin):
#         # Similar to process that happens in IGOR
#         target_numpts = CU.numpts_from_sweeprate(sweeprate, self.measure_freq, start, fin)
#         return round(target_numpts / self.info.wave_len * self.info.num_cycles)
#
#
# @dataclass
# class SquareTransitionModel:
#     mid: Union[float, np.ndarray] = 0.
#     amp: Union[float, np.ndarray] = 0.5
#     theta: Union[float, np.ndarray] = 0.5
#     lin: Union[float, np.ndarray] = 0.01
#     const: Union[float, np.ndarray] = 8.
#
#     square_wave: SquareAWGModel = None
#     cross_cap: Union[float, np.ndarray] = 0.0
#     heat_factor: Union[float, np.ndarray] = 1.5e-6
#     dS: Union[float, np.ndarray] = np.log(2)
#
#     def __post_init__(self):
#         if self.square_wave is None:
#             raise ValueError('Square wave must be passed in to initialize SquareTransitionModel')
#         sw = self.square_wave
#         self.numpts = sw.numpts
#         self.x = sw.x_array
#
#     def eval(self, x, no_heat=False):
#         x = np.asarray(x)
#         if np.any([isinstance(v, np.ndarray) for v in asdict(self).values()]):
#             return self.eval_nd(x)
#         if no_heat is False:
#             heating_v = self.square_wave.eval(x)
#         else:
#             heating_v = np.zeros(x.shape)
#         x = self.get_true_x(x)  # DAC only steps num_steps times
#         z = i_sense_square_heated(x, self.mid, self.theta, self.amp, self.lin, self.const, heating_v,
#                                   self.cross_cap, self.heat_factor, self.dS)
#         return z
#
#     def eval_nd(self, x: np.ndarray):
#         # Note: x is separate from other variables and is used to get heating_v
#
#         # Turn into dictionary so can iterate through values
#         info = asdict(self)
#         heating_v = self.square_wave.eval(x)
#
#         # Get which variables are arrays instead of just values
#         array_keys = []
#         for k, v in info.items():
#             if isinstance(v, np.ndarray):
#                 array_keys.append(k)
#
#         meshes = add_data_dims(*[v for k, v in info.items() if k in array_keys], x)
#
#         # Make meshes into a dict using the keys we got above
#         meshes = {k: v for k, v in zip(array_keys + ['x'], meshes)}
#
#         # Make a list of all of the variables either drawing from meshes, or otherwise just the single values
#         vars = {}
#         for k in list(info.keys())+['x']:
#             vars[k] = meshes[k] if k in meshes else info[k]
#
#         heating_v = match_dims(heating_v, vars['x'], dim=-1)  # x is at last dimension
#         # Evaluate the charge transition at all meshgrid positions in one go (resulting in N+1 dimension array)
#         data_array = i_sense_square_heated(vars['x'], vars['mid'], vars['theta'], vars['amp'], vars['lin'],
#                                            vars['const'], hv=heating_v, cc=vars['cross_cap'],
#                                            hf=vars['heat_factor'], dS=vars['dS'])
#
#         # Add a y dimension to the data so that it is an N+2 dimension array (duplicate all data and then move that axis
#         # to the y position (N, y, x)
#         data2d_array = np.moveaxis(np.repeat([data_array], 2, axis=0), 0, -2)
#         return data2d_array
#
#     def get_true_x(self, x):
#         """
#         Returns the true x_values of the DACs (i.e. taking into account the fact that they only step num_steps times)
#         Args:
#             x (Union(float, np.ndarray)):  x values to evaluate true DAC values at (must be within original x_array to
#             make sense)
#
#         Returns:
#             np.ndarray: True x values with same shape as x passed in (i.e. repeated values where DACs don't change)
#         """
#         true_x = self.square_wave.true_x_array
#         if not np.all([x >= np.nanmin(true_x), x <= np.nanmax(true_x)]):
#             raise ValueError(f'x passed in has min, max = {np.nanmin(x):.1f}, {np.nanmax(x):.1f} which lies outside of '
#                              f'original x_array of model which has min, max = {np.nanmin(true_x):.1f}, '
#                              f'{np.nanmax(true_x):.1f}')
#         x = np.asarray(x)
#         dx = (true_x[-1] - true_x[0]) / true_x.shape[-1]
#         fake_x = np.linspace(true_x[0] + dx / 2, true_x[-1] - dx / 2,
#                              true_x.shape[-1])  # To trick interp nearest to give the
#         # correct values. Tested with short arrays and confirmed this best matches the exact steps (maybe returns wrong
#         # value when asking for value between DAC steps)
#         interper = interp1d(fake_x, true_x, kind='nearest', bounds_error=False,
#                             fill_value='extrapolate')
#         return interper(x)
#
#
# def i_sense_square_heated(x, mid, theta, amp, lin, const, hv, cc, hf, dS):
#     """ Full transition signal with square wave heating and entropy change
#
#     Args:
#         x (Union(float, np.ndarray)):
#         mid (Union(float, np.ndarray)):
#         theta (Union(float, np.ndarray)):
#         amp (Union(float, np.ndarray)):
#         lin (Union(float, np.ndarray)):
#         const (Union(float, np.ndarray)):
#         hv (Union(float, np.ndarray)): Heating Voltage
#         cc (Union(float, np.ndarray)): Cross Capacitance of HV and Plunger gate (i.e. shift middle)
#         hf (Union(float, np.ndarray)): Heat Factor (i.e. how much HV increases theta)
#         dS (Union(float, np.ndarray)): Change in entropy between N -> N+1
#
#     Returns:
#         (Union(float, np.ndarray)): evaluated function at x value(s)
#     """
#     heat = hf * hv ** 2  # Heating proportional to hv^2
#     T = theta + heat  # theta is base temp theta, so T is that plus any heating
#     X = x - mid + dS * heat - hv * cc  # X is x position shifted by entropy when heated and cross capacitance of heating
#     arg = X / (2 * T)
#     return -amp / 2 * np.tanh(arg) + lin * (x - mid) + const  # Linear term is direct interaction with CS
#
# def add_data_dims(*arrays: np.ndarray, squeeze=True):
#     """
#     Adds dimensions to numpy arrays so that they can be broadcast together
#
#     Args:
#         squeeze (bool): Whether to squeeze out dimensions with zero size first
#         *arrays (np.ndarray): Any amount of np.ndarrays to combine together (maintaining order of dimensions passed in)
#
#     Returns:
#         List[np.ndarray]: List of arrays with new broadcastable dimensions
#     """
#     arrays = [arr[:] for arr in arrays]  # So don't alter original arrays
#     if squeeze:
#         arrays = [arr.squeeze() for arr in arrays]
#
#     total_dims = sum(arg.ndim for arg in arrays)  # All args will need these dims by end
#     before = list()
#     for arg in arrays:
#         arg_dims = arg.ndim  # How many dims in current arg (record original value)
#         after = [1] * (total_dims - len(before) - arg_dims)  # How many dims need to be added to end
#         arg.resize(*before, *arg.shape, *after)  # Add dims before and after
#         before += [1] * arg_dims  # Increment dims to add before by number of dims gone through so far
#     return arrays
#
#
# def match_dims(arr, match, dim):
#     """
#     Turns arr into an ndim array which matches 'match' and has values in dimension 'dim'.
#     Useful for broadcasting arrays together, where more than one array should be broadcast together
#
#     Args:
#         arr (np.ndarray): 1D array to turn into ndim array with values at dim
#         match (np.ndarray): The broadcastable arrays to match dims with
#         dim (int): Which dim to move the values to in new array
#
#     Returns:
#         np.ndarray: Array with same ndim as match, and values at dimension dim
#     """
#
#     arr = arr.squeeze()
#     if arr.ndim != 1:
#         raise ValueError(f'new could not be squeezed into a 1D array. Must be 1D')
#
#     if match.shape[dim] != arr.shape[0]:
#         raise ValueError(f'match:{match.shape} at dim:{dim} does not match new shape:{arr.shape}')
#
#     # sparse = np.moveaxis(np.array(arr, ndmin=match.ndim), -1, dim)
#     if dim < 0:
#         dim = match.ndim + dim
#
#     full = np.moveaxis(np.tile(arr, (*match.shape[:dim], *match.shape[dim + 1:], 1)), -1, dim)
#     return full
# endregion





