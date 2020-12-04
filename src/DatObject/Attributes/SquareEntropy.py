from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

import OLD.InDepthData
import src.Plotting.Mpl.PlotUtil
import src.Plotting.Mpl.Plots
from deprecation import deprecated

if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF
from scipy.interpolate import interp1d

from src.DatObject.Attributes.Entropy import *
from src.DatObject.Attributes import AWG, Logs, Transition as T, DatAttribute as DA, Entropy as E
from src.Characters import PM

from dataclasses import dataclass, InitVar, field, asdict, is_dataclass
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# SETTLE_TIME = 1.2e-3  # 9/8/20 -- measured to be ~0.8ms, so using 1.2ms to be safe
SETTLE_TIME = 5e-3  # 9/10/20 -- measured to be ~3.75ms, so using 5ms to be safe (this is with RC low pass filters)


@dataclass(init=False)  # Using to make nice repr etc, but the values will be init from HDF
class SquareWaveAWG(AWG.AWG):
    v0: float
    vp: float
    vm: float

    def __init__(self, hdf):
        super().__init__(hdf)


    def get_from_HDF(self):
        super().get_from_HDF()
        self.ensure_four_setpoints()  # Even if ramps in wave, this will make it look like a 4 point square wave
        square_aw = self.AWs[0]  # Assume AW0 for square wave heating
        if square_aw.shape[-1] == 4:
            self.v0, self.vp, _, self.vm = square_aw[0]
        else:
            logger.warning(f'Unexpected shape of square wave output: {square_aw.shape}')

    def ensure_four_setpoints(self):
        """
        This turns a arbitrary looking wave into 4 main setpoints assuming 4 equal lengths and that the last value in
        each section is the true setpoint.

        This should fix self.get_full_wave_masks() as well

        Assuming that square wave heating is always done with 4 setpoints (even if AWs contain more). This is to allow
        for additional ramp behaviour between setpoints. I.e. 4 main setpoints but with many setpoints to define
        ramps in between.
        """
        if self.AWs is not None:
            new_AWs = list()
            for aw in self.AWs:
                new_AWs.append(_force_four_point_AW(aw))  # Turns any length AW into a 4 point Square AW
            self.AWs = np.array(new_AWs)


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
        len = np.sum(aw_chunk[1])  #
        new_aw[0, i] = sp
        new_aw[1, i] = len
    return new_aw


class SquareEntropy(DA.DatAttribute):
    version = '1.0'
    group_name = 'SquareEntropy'

    def __init__(self, hdf):
        super().__init__(hdf)
        self.x = None
        self.y = None
        self.data = None
        self.Processed: Optional[SquareProcessed] = None
        self.SquareAWG: Optional[SquareWaveAWG] = None
        self.get_from_HDF()

        # For temp storage
        self._entropy_data = None

    @property
    def entropy_data(self):
        if self._entropy_data is None and CU.get_nested_attr_default(self, 'Processed.outputs.cycled', None) is not None:
            data = CU.get_nested_attr_default(self, 'Processed.outputs.cycled', None)
            self._entropy_data = entropy_signal(np.swapaxes(data, 0, 1))  # Put 4 parts as first axis, then returns 2D
        return self._entropy_data

    @property
    def dS(self):
        return self.Processed.outputs.entropy_fit.best_values.dS

    @property
    def ShowPlots(self):
        return self.Processed.plot_info.show

    @ShowPlots.setter
    def ShowPlots(self, value):
        assert isinstance(value, ShowPlots)
        self.Processed.plot_info.show = value

    def get_from_HDF(self):
        super().get_from_HDF()  # Doesn't do much
        dg = self.group.get('Data', None)
        if dg is not None:
            self.x = dg.get('x', None)
            self.y = dg.get('y', None)
            if isinstance(self.y, float) and np.isnan(self.y):  # Because I store None as np.nan
                self.y = None
            self.data = dg.get('i_sense', None)
        self.Processed = self._get_square_processed()
        self.ShowPlots = self.Processed.plot_info.show
        self.SquareAWG = SquareWaveAWG(self.hdf)

    def _get_square_processed(self):
        awg = SquareWaveAWG(self.hdf)
        inp = Input(self.data, self.x, awg, bias=None, transition_amplitude=None)
        spg = self.group.get('SquareProcessed')
        if spg is not None:
            sp_data = dict()
            sp_data['Input'] = inp
            for name, dc, sdc in zip(['ProcessParams', 'Outputs', 'PlotInfo'], [ProcessParams, Output, PlotInfo], [{}, {'integrated_info': IntegratedInfo}, {'show': ShowPlots}]):
                g = spg.get(name)
                if g is not None:
                    sp_data[name] = dataclass_from_group(g, dc=dc, sub_dataclass=sdc)
            # Make dict with correct keys for SquareProcessed
            spdata = {k: sp_data.pop(o_k) for k, o_k in zip(['inputs', 'process_params', 'outputs', 'plot_info'],
                                                            ['Input', 'ProcessParams', 'Outputs', 'PlotInfo'])}
            sp = SquareProcessed(**spdata)
        else:
            sp = SquareProcessed(process_params=ProcessParams(setpoint_start=int(np.round(SETTLE_TIME*awg.measure_freq))))
        return sp

    def update_HDF(self):
        super().update_HDF()
        # self.group.attrs['description'] =
        dg = self.group.require_group('Data')
        for name, data in zip(['x', 'y', 'i_sense'], [self.x, self.y, self.data]):
            if data is None:
                data = np.nan
            HDU.set_data(dg, name, data)  # Removes dataset before setting if necessary
        self._set_square_processed()
        self.hdf.flush()

    def _set_square_processed(self):
        sp = self.Processed
        spg = self.group.require_group('SquareProcessed')
        # inpg = spg.require_group('Inputs')  # Only drawn from HDF data anyway
        ppg = spg.require_group('ProcessParams')
        outg = spg.require_group('Outputs')
        pig = spg.require_group('PlotInfo')
        # dataclass_to_group(inpg, sp.inputs)
        dataclass_to_group(ppg, sp.process_params)
        dataclass_to_group(outg, sp.outputs)
        dataclass_to_group(pig, sp.plot_info, ignore=['axs', 'axs_dict'])

    def _check_default_group_attrs(self):
        super()._check_default_group_attrs()

    def process(self):
        awg = SquareWaveAWG(self.hdf)
        # transition = T.NewTransitions(self.hdf)
        assert awg is not None
        # assert transition is not None
        sp = self.Processed if self.Processed else SquareProcessed()

        inp = sp.inputs
        if None in [inp.raw_data, inp.orig_x_array, inp.awg]: # Re init if necessary
            inp = Input(raw_data=self.data, orig_x_array=self.x, awg=awg, bias=None, transition_amplitude=None)

        # Use already stored process_params (which default to reasonable settings anyway)
        pp = sp.process_params

        # Recalculate ouptuts
        out = process(inp, pp)

        # Keep same plot_info as previous (or default)
        plot_info = sp.plot_info

        sp = SquareProcessed(inp, pp, out, plot_info)
        self.Processed = sp
        self.update_HDF()


def dataclass_to_group(group: h5py.Group, dc, ignore=None):
    """
    Stores all values from dataclass into group, can be used to re init the given dataclass later
    Args:
        group (h5py.Group):
        dc (dataclass):
        ignore (Union(List[str], str)): Any Dataclass entries not to store (definitely anything that is not in init!!)
    Returns:
        (None):
    """
    ignore = ignore if ignore else list()
    dc_path = '.'.join((dc.__class__.__module__, dc.__class__.__name__))
    HDU.set_attr(group, 'Dataclass', dc_path)
    group.attrs['description'] = 'Dataclass'
    assert is_dataclass(dc)
    # for k, v in asdict(dc).items():  # Tries to be too clever about inner attributes (e.g. lm.Parameters doesn't work)
    for k in dc.__annotations__:
        v = getattr(dc, k)
        if k not in ignore:
            if isinstance(v, (np.ndarray, h5py.Dataset)):
                HDU.set_data(group, k, v)
            elif isinstance(v, list):
                HDU.set_list(group, k, v)
            elif isinstance(v, DA.FitInfo):
                v.save_to_hdf(group, k)
            elif is_dataclass(sub_dc := getattr(dc, k)):
                sub_group = group.require_group(k)
                dataclass_to_group(sub_group, sub_dc)
            else:
                HDU.set_attr(group, k, v)


def dataclass_from_group(group, dc, sub_dataclass=None):
    """
    Restores dataclass from HDF
    Args:
       group (h5py.Group):
        dataclass (dataclass):
        sub_dataclass (Optional[dict]): Dict of key: Dataclass for any sub_dataclasses

    Returns:
        (dataclass): Returns filled dataclass instance
    """
    assert group.attrs.get('description') == 'Dataclass'

    all_keys = set(group.keys()).union(set(group.attrs.keys())) - {'Dataclass', 'description'}

    d = dict()
    for k in all_keys:
        v = HDU.get_attr(group, k)
        if v is None:  # For loading datasets, and if it really doesn't exist, None will be returned again
            v = group.get(k, None)
            if isinstance(v, h5py.Group):
                description = v.attrs.get('description')
                if description == 'list':
                    v = HDU.get_list(group, k)
                elif description == 'FitInfo':
                    v = DA.fit_group_to_FitInfo(v)
                elif description == 'Dataclass':
                    if k in sub_dataclass:
                        v = dataclass_from_group(v, sub_dataclass[k])
                    else:
                        raise KeyError(f'Trying to load a dataclass [{k}] without specifying the Dataclass type. Please provide in sub_dataclass')
                elif description is None and not v.keys() and not v.attrs.keys():  # Empty Group... Nothing saved in it
                    v = None
                else:
                    logger.warning(f'{v} is unexpected entry which seems to contain some data. None returned')
                    v = None
            elif isinstance(v, h5py.Dataset):
                v = v[:]
        d[k] = v
    initialized_dc: dataclass = dc(**d)
    return initialized_dc


# region Modelling Only
""" Override SquareWaveAWG class to allow it to be created as a model
Also includes modelling function in this section"""


@dataclass
class SquareAWGModel(SquareWaveAWG):
    measure_freq: float = 1000
    start: InitVar[float] = -10
    fin: InitVar[float] = 10
    sweeprate: InitVar[float] = 1

    v0: float = 0
    vp: float = 100
    vm: float = -100

    step_duration: InitVar[float] = 0.25
    _step_dur: Union[float, None] = field(default=None, repr=False, init=False)

    def __post_init__(self, start: float, fin: float, sweeprate: float, step_duration: float):
        self.step_dur = step_duration
        self.num_steps = None  # Because self.info is called in get_numsteps and includes num_steps
        self.num_steps = self._get_numsteps(sweeprate, start, fin)
        self.x_array = np.linspace(start, fin, self.numpts)

    @property
    def step_dur(self):
        return self._step_dur

    @step_dur.setter
    def step_dur(self, value):
        self._step_dur = round(value * self.measure_freq) / self.measure_freq

    @property
    def info(self):
        wave_len = self.step_dur * self.measure_freq * 4
        assert np.isclose(round(wave_len), wave_len, atol=0.00001)  # Should be an int
        return Logs.AWGtuple(outputs={0: [0]},  # wave 0 output 0
                             wave_len=int(wave_len),
                             num_adcs=1,
                             samplingFreq=self.measure_freq,
                             measureFreq=self.measure_freq,
                             num_cycles=1,
                             num_steps=self.num_steps)

    @property
    def AWs(self):
        step_samples = round(self.step_dur * self.measure_freq)
        assert np.isclose(round(step_samples), step_samples, atol=0.00001)  # Should be an int
        return [np.array([[self.v0, self.vp, self.v0, self.vm],
                          [int(step_samples)] * 4])]

    def _get_numsteps(self, sweeprate, start, fin):
        # Similar to process that happens in IGOR
        target_numpts = CU.numpts_from_sweeprate(sweeprate, self.measure_freq, start, fin)
        return round(target_numpts / self.info.wave_len * self.info.num_cycles)


@dataclass
class SquareTransitionModel:
    mid: Union[float, np.ndarray] = 0.
    amp: Union[float, np.ndarray] = 0.5
    theta: Union[float, np.ndarray] = 0.5
    lin: Union[float, np.ndarray] = 0.01
    const: Union[float, np.ndarray] = 8.

    square_wave: SquareAWGModel = None
    cross_cap: Union[float, np.ndarray] = 0.0
    heat_factor: Union[float, np.ndarray] = 1.5e-6
    dS: Union[float, np.ndarray] = np.log(2)

    def __post_init__(self):
        if self.square_wave is None:
            raise ValueError('Square wave must be passed in to initialize SquareTransitionModel')
        sw = self.square_wave
        self.numpts = sw.numpts
        self.x = sw.x_array

    def eval(self, x, no_heat=False):
        x = np.asarray(x)
        if np.any([isinstance(v, np.ndarray) for v in asdict(self).values()]):
            return self.eval_nd(x)
        if no_heat is False:
            heating_v = self.square_wave.eval(x)
        else:
            heating_v = np.zeros(x.shape)
        x = self.get_true_x(x)  # DAC only steps num_steps times
        z = i_sense_square_heated(x, self.mid, self.theta, self.amp, self.lin, self.const, heating_v,
                                  self.cross_cap, self.heat_factor, self.dS)
        return z

    def eval_nd(self, x: np.ndarray):
        # Note: x is separate from other variables and is used to get heating_v

        # Turn into dictionary so can iterate through values
        info = asdict(self)
        heating_v = self.square_wave.eval(x)

        # Get which variables are arrays instead of just values
        array_keys = []
        for k, v in info.items():
            if isinstance(v, np.ndarray):
                array_keys.append(k)

        meshes = add_data_dims(*[v for k, v in info.items() if k in array_keys], x)

        # Make meshes into a dict using the keys we got above
        meshes = {k: v for k, v in zip(array_keys + ['x'], meshes)}

        # Make a list of all of the variables either drawing from meshes, or otherwise just the single values
        vars = {}
        for k in list(info.keys())+['x']:
            vars[k] = meshes[k] if k in meshes else info[k]

        heating_v = match_dims(heating_v, vars['x'], dim=-1)  # x is at last dimension
        # Evaluate the charge transition at all meshgrid positions in one go (resulting in N+1 dimension array)
        data_array = i_sense_square_heated(vars['x'], vars['mid'], vars['theta'], vars['amp'], vars['lin'],
                                           vars['const'], hv=heating_v, cc=vars['cross_cap'],
                                           hf=vars['heat_factor'], dS=vars['dS'])

        # Add a y dimension to the data so that it is an N+2 dimension array (duplicate all data and then move that axis
        # to the y position (N, y, x)
        data2d_array = np.moveaxis(np.repeat([data_array], 2, axis=0), 0, -2)
        return data2d_array

    def get_true_x(self, x):
        """
        Returns the true x_values of the DACs (i.e. taking into account the fact that they only step num_steps times)
        Args:
            x (Union(float, np.ndarray)):  x values to evaluate true DAC values at (must be within original x_array to
            make sense)

        Returns:
            np.ndarray: True x values with same shape as x passed in (i.e. repeated values where DACs don't change)
        """
        true_x = self.square_wave.true_x_array
        if not np.all([x >= np.nanmin(true_x), x <= np.nanmax(true_x)]):
            raise ValueError(f'x passed in has min, max = {np.nanmin(x):.1f}, {np.nanmax(x):.1f} which lies outside of '
                             f'original x_array of model which has min, max = {np.nanmin(true_x):.1f}, '
                             f'{np.nanmax(true_x):.1f}')
        x = np.asarray(x)
        dx = (true_x[-1] - true_x[0]) / true_x.shape[-1]
        fake_x = np.linspace(true_x[0] + dx / 2, true_x[-1] - dx / 2,
                             true_x.shape[-1])  # To trick interp nearest to give the
        # correct values. Tested with short arrays and confirmed this best matches the exact steps (maybe returns wrong
        # value when asking for value between DAC steps)
        interper = interp1d(fake_x, true_x, kind='nearest', bounds_error=False,
                            fill_value='extrapolate')
        return interper(x)


def i_sense_square_heated(x, mid, theta, amp, lin, const, hv, cc, hf, dS):
    """ Full transition signal with square wave heating and entropy change

    Args:
        x (Union(float, np.ndarray)):
        mid (Union(float, np.ndarray)):
        theta (Union(float, np.ndarray)):
        amp (Union(float, np.ndarray)):
        lin (Union(float, np.ndarray)):
        const (Union(float, np.ndarray)):
        hv (Union(float, np.ndarray)): Heating Voltage
        cc (Union(float, np.ndarray)): Cross Capacitance of HV and Plunger gate (i.e. shift middle)
        hf (Union(float, np.ndarray)): Heat Factor (i.e. how much HV increases theta)
        dS (Union(float, np.ndarray)): Change in entropy between N -> N+1

    Returns:
        (Union(float, np.ndarray)): evaluated function at x value(s)
    """
    heat = hf * hv ** 2  # Heating proportional to hv^2
    T = theta + heat  # theta is base temp theta, so T is that plus any heating
    X = x - mid + dS * heat - hv * cc  # X is x position shifted by entropy when heated and cross capacitance of heating
    arg = X / (2 * T)
    return -amp / 2 * np.tanh(arg) + lin * (x - mid) + const  # Linear term is direct interaction with CS


# endregion


# region Processing functions from I_sense to Entropy
"""All the functions for processing I_sense data into the various steps of square wave heated data"""


def chunk_data(data, awg: SquareWaveAWG):
    """
    Breaks up data into chunks which make more sense for square wave heating datasets.
    Args:
        data (np.ndarray): 1D or 2D data (full data to match original x_array).
            Note: will return with y dim regardless of 1D or 2D
        awg (SquareWaveAWG): AWG part of dat which has all square wave info in.

    Returns:
        List[np.ndarray]: Data broken up into chunks (setpoints, (ylen, num_steps, num_cycles, sp_len)).

            NOTE: Has to be a list returned and not a ndarray because sp_len may vary per steps

            NOTE: This is the only step where setpoints should come first, once sp_len binned it should be ylen first
    """
    wave_num = 0
    masks = awg.get_full_wave_masks(wave_num)
    AW = awg.AWs[wave_num]  # [[setpoints],[lengths]]
    num_steps = awg.info.num_steps
    num_cycles = awg.info.num_cycles
    zs = []
    for mask, sp_len in zip(masks, AW[1].astype(int)):
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


def average_2D(x, data, centers=None, avg_nans=False):
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
            fits = T.transition_fits(x, z0_avg_per_row)
            if np.any([fit is None for fit in fits]):  # Not able to do transition fits for some reason
                logger.warning(f'{np.sum([1 if fit is None else 0 for fit in fits])} transition fits failed, blind '
                               f'averaging instead of centered averaging')
                return x, np.mean(data, axis=0)
            fit_infos = [DA.FitInfo.from_fit(fit) for fit in fits]  # Has my functions associated
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
    return nx, ndata


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


# region Processing Info
@dataclass
class IntegratedInfo:
    # Inputs
    dT: int = None
    amp: float = None
    dx: float = None

    # Calculated
    dS: float = None

    @property
    def sf(self):
        return scaling(self.dT, self.amp, self.dx)


@dataclass
class Input:
    # Attrs that will be set from dat if possible
    raw_data: np.ndarray = field(default=None, repr=False)  # basic 1D or 2D data (Needs to match original x_array
    # for awg)
    orig_x_array: np.ndarray = field(default=None, repr=False)  # original x_array (needs to be original for awg)
    awg: SquareWaveAWG = field(default=None, repr=True)  # AWG class from dat for chunking data AND for plot_info
    datnum: Optional[int] = field(default=None, repr=True)  # For plot_info plot title
    x_label: Optional[str] = field(default=None, repr=True)  # For plots
    bias: Optional[float] = field(default=None, repr=True)  # Square wave bias applied (assumed symmetrical since
    # only averaging for now anyway)
    transition_amplitude: Optional[float] = field(default=None, repr=True)  # For calculating scaling factor for
    # integrated entropy
    dT: Optional[float] = field(default=None, repr=True)  # Can directly supply dT instead of bias for integrated
    # entropy calculation

@dataclass
class ProcessParams:
    setpoint_start: Optional[int] = None  # Index to start averaging for each setpoint
    setpoint_fin: Optional[int] = None  # Index to stop averaging for each setpoint
    cycle_start: Optional[int] = None  # Index to start averaging cycles
    cycle_fin: Optional[int] = None  # Index to stop averaging cycles

    # Integrated Params
    bias_theta_lookup: dict = field(
        default_factory=lambda: {0: 0.4942, 30: 0.7608, 50: 1.0301, 80: 1.4497})  # Bias/nA: Theta/mV


@dataclass
class Output:
    # Data that will be calculated
    x: np.ndarray = field(default=None, repr=False)  # x_array with length of num_steps (for cycled, averaged, entropy)
    chunked: np.ndarray = field(default=None, repr=False)  # Data broken in to chunks based on AWG (just plot
    # raw_data on orig_x_array)
    setpoint_averaged: np.ndarray = field(default=None, repr=False)  # Setpoints averaged only
    setpoint_averaged_x: np.ndarray = field(default=None, repr=False)  # x_array for setpoints averaged only
    cycled: np.ndarray = field(default=None, repr=False)  # setpoint averaged and then cycles averaged data
    averaged: np.ndarray = field(default=None, repr=False)  # setpoint averaged, cycle_avg, then averaged in y
    entropy_signal: np.ndarray = field(default=None, repr=False)  # Entropy signal data (same x as averaged data)
    integrated_entropy: np.ndarray = field(default=None, repr=False)  # Integrated entropy signal (same x as averaged
    # data)

    entropy_fit: DA.FitInfo = field(default=None, repr=True)  # FitInfo of entropy fit to self.entropy_signal
    integrated_info: IntegratedInfo = field(default=None, repr=True)  # Things like dt, sf, amp etc


@dataclass
class ShowPlots:
    info: bool = False
    raw: bool = False
    setpoint_averaged: bool = False
    cycle_averaged: bool = False
    averaged: bool = False
    entropy: bool = False
    integrated: bool = False


@dataclass
class PlotInfo:
    axs: Union[np.ndarray[plt.Axes], None] = field(default=None, repr=False)
    show: ShowPlots = ShowPlots()
    decimate_freq: float = 50  # Target decimation freq when plotting raw. Set to None for no decimation
    plot_row_num: int = 0  # Which row of data to plot where this is an option
    axs_dict: dict = field(init=False, repr=False)

    def __post_init__(self):
        if self.axs is not None:
            self.axs_dict = {k: v for k, v in zip(asdict(self.show).keys(), self.axs)}
        else:
            self.axs_dict = {k: None for k in asdict(self.show).keys()}

    @property
    def num_plots(self):
        return sum(asdict(self.show).values())


@dataclass
class SquareProcessed:
    inputs: Input = Input()
    process_params: ProcessParams = ProcessParams()
    outputs: Output = Output()
    plot_info: PlotInfo = PlotInfo()

    @classmethod
    @deprecated
    def from_dat(cls, dat: DatHDF, calculate=True):
        """
        Extracts data necessary for square wave analysis from datHDF object.
        Args:
            dat (DatHDF):
            calculate (bool):

        Returns:

        """

        inp = Input(dat.Data.i_sense, dat.Data.x_array, dat.AWG, dat.datnum, dat.Logs.x_label,
                    dat.AWG.AWs[0][0][1]/10, dat.Transition.avg_fit.best_values.amp)
        pp = ProcessParams()
        if calculate:
            out = process(inp, pp)
        else:
            out = Output()
        plot_info = PlotInfo()
        return cls(inp, pp, out, plot_info)

    @classmethod
    def from_info(cls, data, x, awg, bias=None, amplitude=None, datnum=None, x_label=None, calculate=True):
        """
        Convenient function to help do square wave entropy processing on non Dat data.
        Args:
            data (np.ndarray): 1D or 2D I_sense data which has to match the AWG info
            x (np.ndarray): original x_array which matches awg
            awg (AWG.AWG): AWG attribute
            bias (float): Heating bias applied in nA
            amplitude (float): Charge transition amplitude for Integrated entropy calculation
            datnum (Union(int, None)): Use for plot titles etc
            x_label (str): Used for plots
            calculate (bool):  Whether to run processing straight away. Set to False if want to change ProcessParams

        Returns:
            SquareProcessed: Full SquareProcessed dataclass which can be used to plot along with PlotInfo
        """
        inp = Input(data, x, awg, datnum, x_label, bias, amplitude)
        pp = ProcessParams()
        if calculate:
            out = process(inp, pp)
        else:
            out = Output()
        plot_info = PlotInfo()
        return cls(inp, pp, out, plot_info)

    def calculate(self):
        self.outputs = process(self.inputs, self.process_params)


# endregion


def process(input_info: Input, process_pars: ProcessParams) -> Output:
    output = Output()
    inp = input_info
    pp = process_pars

    # Calculate true x_array (num_steps)
    output.x = np.linspace(inp.orig_x_array[0], inp.orig_x_array[-1], inp.awg.info.num_steps)

    # Get chunked data (setpoints, ylen, numsteps, numcycles, splen)
    output.chunked = chunk_data(inp.raw_data, inp.awg)

    # Average setpoints of data ([ylen], setpoints, numsteps, numcycles)
    output.setpoint_averaged = average_setpoints(output.chunked, start_index=pp.setpoint_start,
                                                 fin_index=pp.setpoint_fin)
    output.setpoint_averaged_x = np.linspace(inp.orig_x_array[0], inp.orig_x_array[-1],
                                             inp.awg.info.num_steps * inp.awg.info.num_cycles)

    # Averaged cycles ([ylen], setpoints, numsteps)
    output.cycled = average_cycles(output.setpoint_averaged, start_cycle=pp.cycle_start, fin_cycle=pp.cycle_fin)

    # Center and average 2D data or skip for 1D
    output.x, output.averaged = average_2D(output.x, output.cycled)

    # region Use this if want to start shifting each heater setpoint of data left or right
    # Align data
    # output.x, output.averaged = align_setpoint_data(xs, output.averaged, nx=None)
    # endregion

    # Entropy signal
    output.entropy_signal = entropy_signal(output.averaged)

    # Fit Entropy
    ent_fit = E.entropy_fits(output.x, output.entropy_signal)[0]
    output.entropy_fit = DA.FitInfo.from_fit(ent_fit)

    # Integrate Entropy
    dx = float(np.mean(np.diff(output.x)))
    if inp.bias is not None and inp.dT is not None:
        logger.warning(f'inp.bias = {inp.bias} and inp.dT = {inp.dT} both supplied. ONLY using inp.dT')
    if inp.dT is not None:
        dT = inp.dT
    elif inp.bias in pp.bias_theta_lookup:
        dT = pp.bias_theta_lookup[inp.bias] - pp.bias_theta_lookup[0]
    elif inp.bias is not None:
        raise ValueError(f'inp.bias = {inp.bias} passed, but not found in pp.bias_theta_lookup = {pp.bias_theta_lookup}')
    else:
        dT = None

    amp = inp.transition_amplitude
    dx = dx
    if None not in [dT, amp, dx]:
        int_info = IntegratedInfo(dT=dT, amp=inp.transition_amplitude, dx=dx)
        output.integrated_entropy = integrate_entropy(output.entropy_signal, int_info.sf)
        int_info.dS = output.integrated_entropy[-1]
        output.integrated_info = int_info
    else:
        pass
    return output


class Plot:
    """
    All the plotting functions used in plot_square_entropy so they can be easily accessed on an individual basis
    """

    @staticmethod
    def info(awg, ax=None, datnum=None):
        """
        Mostly adds AWG info to an axes
        Args:
            awg (AWG.AWG):  AWG instance
            ax (plt.Axes): Optional axes to plot on
            datnum (int): Optional datnum to add as title

        Returns:
            plt.Axes: The axes that were plotted on
        """
        ax = src.Plotting.Mpl.PlotUtil.require_axs(1, ax, clear=True)[0]
        freq = awg.info.measureFreq / awg.info.wave_len
        sqw_info = f'Square Wave:\n' \
                   f'Output amplitude: {int(awg.AWs[0][0][1]):d}mV\n' \
                   f'Heating current: {awg.AWs[0][0][1] / 10:.1f}nA\n' \
                   f'Frequency: {freq:.1f}Hz\n' \
                   f'Measure Freq: {awg.info.measureFreq:.1f}Hz\n' \
                   f'Num Steps: {awg.info.num_steps:d}\n' \
                   f'Num Cycles: {awg.info.num_cycles:d}\n' \
                   f'SP samples: {int(awg.AWs[0][1][0]):d}\n'

        scan_info = f'Scan info:\n' \
                    f'Sweeprate: {CU.get_sweeprate(awg.info.measureFreq, awg.x_array):.2f}mV/s\n'

        src.Plotting.Mpl.PlotUtil.ax_text(ax, sqw_info, loc=(0.05, 0.05), fontsize=7)
        src.Plotting.Mpl.PlotUtil.ax_text(ax, scan_info, loc=(0.5, 0.05), fontsize=7)
        if datnum:
            src.Plotting.Mpl.PlotUtil.ax_setup(ax, f'Dat{datnum}')
        ax.axis('off')
        return ax

    @staticmethod
    def raw(x, data, ax=None, decimate_freq=None, measure_freq=None, clear=True):
        ax = src.Plotting.Mpl.PlotUtil.require_axs(1, ax, clear=clear)[0]
        if decimate_freq is not None:
            assert measure_freq is not None
            data, freq = CU.decimate(data, measure_freq, decimate_freq, return_freq=True)
            x = np.linspace(x[0], x[-1], data.shape[-1])
            title = f'CS data:\nDecimated to ~{freq:.1f}/s'
        else:
            title = f'CS data'
        src.Plotting.Mpl.Plots.display_1d(x, data, ax, linewidth=1, marker='', auto_bin=False)
        src.Plotting.Mpl.PlotUtil.ax_setup(ax, title)
        return ax

    @staticmethod
    def setpoint_averaged(x, setpoints_data, ax=None, clear=True):
        ax = src.Plotting.Mpl.PlotUtil.require_axs(1, ax, clear=clear)[0]
        for z, label in zip(setpoints_data, ['v0_1', 'vp', 'v0_2', 'vm']):
            ax.plot(x, z.flatten(), label=label)
        src.Plotting.Mpl.PlotUtil.ax_setup(ax, f'Setpoint averaged', legend=True)
        return ax

    @staticmethod
    def cycle_averaged(x, cycle_data, ax=None, clear=True):
        ax = src.Plotting.Mpl.PlotUtil.require_axs(1, ax, clear=clear)[0]
        for z, label in zip(cycle_data, ['v0_1', 'vp', 'v0_2', 'vm']):
            ax.plot(x, z, label=label)
        src.Plotting.Mpl.PlotUtil.ax_setup(ax, f'Setpoint and cycles averaged', legend=True)
        return ax

    @staticmethod
    def averaged(x, averaged_data, ax=None, clear=True):
        ax = src.Plotting.Mpl.PlotUtil.require_axs(1, ax, clear=clear)[0]
        for z, label in zip(averaged_data, ['v0_1', 'vp', 'v0_2', 'vm']):
            ax.plot(x, z, label=label, marker='+')
        src.Plotting.Mpl.PlotUtil.ax_setup(ax, f'Centered with v0\nthen averaged data', legend=True)
        return ax

    @staticmethod
    def entropy(x, entropy_data, entropy_fit, ax=None, clear=True):
        """

        Args:
            x (np.ndarray):
            entropy_data (np.ndarray):
            entropy_fit (DA.FitInfo):
            ax (plt.Axes):
            clear (bool):

        Returns:
            plt.Axes:
        """
        ax = src.Plotting.Mpl.PlotUtil.require_axs(1, ax, clear=clear)[0]
        ax.plot(x, entropy_data, label=f'data')
        temp_x = np.linspace(x[0], x[-1], 1000)
        ax.plot(temp_x, entropy_fit.eval_fit(temp_x), label='fit')
        src.Plotting.Mpl.PlotUtil.ax_setup(ax, f'Entropy: dS = {entropy_fit.best_values.dS:.2f}'
                        f'{PM}{entropy_fit.params["dS"].stderr:.2f}', legend=True)
        return ax

    @staticmethod
    def integrated(x, integrated_data, integrated_info, ax=None, clear=True):
        """

        Args:
            x (np.ndarray):
            integrated_data (np.ndarray):
            integrated_info (IntegratedInfo):
            ax (plt.Axes):
            clear (bool):

        Returns:
            plt.Axes:
        """
        ax = src.Plotting.Mpl.PlotUtil.require_axs(1, ax, clear=clear)[0]
        ax.plot(x, integrated_data)
        ax.axhline(np.log(2), color='k', linestyle=':', label='Ln2')
        src.Plotting.Mpl.PlotUtil.ax_setup(ax, f'Integrated Entropy:\ndS = {integrated_info.dS:.2f}, '
                        f'SF={integrated_info.sf:.3f}', y_label='Entropy /kB', legend=True)
        return ax


def plot_square_entropy(data, sub_poly=True):
    """
    All relevant plots for SquareProcessed data. Axes are updated in plot_info
    Args:
        data (Union[SquareProcessed, DatHDF]): Either SquareProcessed data (including inputs, process_params, outputs),
            or a dat instance which includes dat.SquareEntropy.Processed
            Note: sp.PlotInfo contains a couple of parameters which can be changed
            e.g. decimate_freq, row_num, and also stores the axes plotted on

    Returns:
        None: Nothing from the function itself, axes stored in sp.plot_info
    """

    def next_ax(name) -> plt.Axes:
        nonlocal ax_index, axs, axs_dict
        ax = axs[ax_index]
        ax_index += 1
        if name in axs_dict.keys():
            axs_dict[name] = ax
        else:
            raise ValueError(f'{name} not in plot_info.axs_dict.keys() = {axs_dict.keys()}')
        return ax

    if isinstance(data, SquareProcessed):
        sp = data
        datnum = None
    elif hasattr(data, 'SquareEntropy') and hasattr(data.SquareEntropy, 'Processed'):
        sp = data.SquareEntropy.Processed
        datnum = getattr(data, 'datnum', None)
    else:
        raise TypeError(f'input data does not have the right attributes, type of data is {type(data)}')

    plot_info = sp.plot_info
    plot_info.axs = np.atleast_1d(src.Plotting.Mpl.PlotUtil.require_axs(plot_info.num_plots, plot_info.axs, clear=True))

    if plot_info.axs[0].figure._suptitle is None and datnum is not None:
        fig = plot_info.axs[0].figure
        fig.suptitle(f'Dat{datnum}')

    ax_index = 0
    axs = plot_info.axs
    show = plot_info.show
    axs_dict = plot_info.axs_dict

    for ax in axs:
        ax.set_xlabel(sp.inputs.x_label)
        ax.set_ylabel('Current /nA')

    if show.info:
        ax = next_ax('info')
        awg = sp.inputs.awg
        Plot.info(awg, ax, sp.inputs.datnum)

    if show.raw:
        ax = next_ax('raw')
        z = sp.inputs.raw_data[plot_info.plot_row_num]
        x = sp.inputs.orig_x_array
        Plot.raw(x, z, ax, plot_info.decimate_freq, sp.inputs.awg.measure_freq)
        src.Plotting.Mpl.PlotUtil.edit_title(ax, f'Row {plot_info.plot_row_num} of ', prepend=True)

    if show.setpoint_averaged:
        ax = next_ax('setpoint_averaged')
        Plot.setpoint_averaged(sp.outputs.setpoint_averaged_x, sp.outputs.setpoint_averaged[plot_info.plot_row_num], ax, clear=True)
        src.Plotting.Mpl.PlotUtil.edit_title(ax, f'Row {plot_info.plot_row_num} of ', prepend=True)

    if show.cycle_averaged:
        ax = next_ax('cycle_averaged')
        Plot.cycle_averaged(sp.outputs.x, sp.outputs.cycled[plot_info.plot_row_num], ax, clear=True)
        src.Plotting.Mpl.PlotUtil.edit_title(ax, f'Row {plot_info.plot_row_num} of ', prepend=True)

    if show.averaged:
        ax = next_ax('averaged')
        x = sp.outputs.x
        z = sp.outputs.averaged
        if sub_poly is True:
            fit = T.transition_fits(x, z[0], func=T.i_sense_digamma_quad)[0]
            x, z = OLD.InDepthData.sub_poly_from_data(x, z, fit)
        Plot.averaged(x, z, ax, clear=True)
        # Plot.averaged(sp.outputs.x, sp.outputs.averaged, ax, clear=True)

    if show.entropy:
        ax = next_ax('entropy')
        Plot.entropy(sp.outputs.x, sp.outputs.entropy_signal, sp.outputs.entropy_fit, ax, clear=True)

    if show.integrated:
        ax = next_ax('integrated')
        Plot.integrated(sp.outputs.x, sp.outputs.integrated_entropy, sp.outputs.integrated_info, ax, clear=True)

    axs[0].get_figure().tight_layout()
    return


def add_data_dims(*arrays: np.ndarray, squeeze=True):
    """
    Adds dimensions to numpy arrays so that they can be broadcast together

    Args:
        squeeze (bool): Whether to squeeze out dimensions with zero size first
        *arrays (np.ndarray): Any amount of np.ndarrays to combine together (maintaining order of dimensions passed in)

    Returns:
        List[np.ndarray]: List of arrays with new broadcastable dimensions
    """
    arrays = [arr[:] for arr in arrays]  # So don't alter original arrays
    if squeeze:
        arrays = [arr.squeeze() for arr in arrays]

    total_dims = sum(arg.ndim for arg in arrays)  # All args will need these dims by end
    before = list()
    for arg in arrays:
        arg_dims = arg.ndim  # How many dims in current arg (record original value)
        after = [1] * (total_dims - len(before) - arg_dims)  # How many dims need to be added to end
        arg.resize(*before, *arg.shape, *after)  # Add dims before and after
        before += [1] * arg_dims  # Increment dims to add before by number of dims gone through so far
    return arrays


def match_dims(arr, match, dim):
    """
    Turns arr into an ndim array which matches 'match' and has values in dimension 'dim'.
    Useful for broadcasting arrays together, where more than one array should be broadcast together

    Args:
        arr (np.ndarray): 1D array to turn into ndim array with values at dim
        match (np.ndarray): The broadcastable arrays to match dims with
        dim (int): Which dim to move the values to in new array

    Returns:
        np.ndarray: Array with same ndim as match, and values at dimension dim
    """

    arr = arr.squeeze()
    if arr.ndim != 1:
        raise ValueError(f'new could not be squeezed into a 1D array. Must be 1D')

    if match.shape[dim] != arr.shape[0]:
        raise ValueError(f'match:{match.shape} at dim:{dim} does not match new shape:{arr.shape}')

    # sparse = np.moveaxis(np.array(arr, ndmin=match.ndim), -1, dim)
    if dim < 0:
        dim = match.ndim + dim

    full = np.moveaxis(np.tile(arr, (*match.shape[:dim], *match.shape[dim + 1:], 1)), -1, dim)
    return full