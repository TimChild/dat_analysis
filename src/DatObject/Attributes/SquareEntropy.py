import numpy
from scipy.interpolate import interp1d

from src import CoreUtil
from src.DatObject.Attributes.Entropy import *
from src.DatObject.Attributes import AWG, Logs, Transition as T, DatAttribute as DA
from dataclasses import dataclass, InitVar, field, _MISSING_TYPE

from src.Scripts.StandardImports import logger


@dataclass(init=False)  # Using to make nice repr etc, but the values will be init from HDF
class SquareWaveAWG(AWG.AWG):
    v0: float
    vp: float
    vm: float

    def __init__(self, hdf):
        super().__init__(hdf)
        self.get_from_HDF()

    def get_from_HDF(self):
        super().get_from_HDF()
        square_aw = self.AWs[0]  # Assume AW0 for square wave heating
        if square_aw.shape[-1] == 4:
            self.v0, self.vp, _, self.vm = square_aw[0]
        else:
            logger.warning(f'Unexpected shape of square wave output: {square_aw.shape}')


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
    _step_dur: Union[float, None] = field(default=None, repr=False)

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
        print('setting')
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
    mid: float = 0.
    amp: float = 0.5
    theta: float = 0.5
    lin: float = 0.01
    const: float = 8.

    square_wave: SquareAWGModel = None
    cross_cap: float = 0.0
    heat_factor: float = 1.5e-6
    dS: float = np.log(2)

    def __post_init__(self):
        if self.square_wave is None:
            raise ValueError('Square wave must be passed in to initialize SquareTransitionModel')
        sw = self.square_wave
        self.numpts = sw.numpts
        self.x = sw.x_array

    def eval(self, x):
        x = np.asarray(x)
        heating_v = self.square_wave.eval(x)
        x = self.get_true_x(x)  # DAC only steps num_steps times
        z = i_sense_square_heated(x, self.mid, self.theta, self.amp, self.lin, self.const, heating_v,
                                  self.cross_cap, self.heat_factor, self.dS)
        return z

    def get_true_x(self, x):
        """
        Returns the true x_values of the DACs (i.e. taking into account the fact that they only step num_steps times)
        Args:
            x (Union[float, np.ndarray]):  x values to evaluate true DAC values at (must be within original x_array to
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
        fake_x = np.linspace(true_x[0]+dx/2, true_x[-1]-dx/2, true_x.shape[-1])  # To trick interp nearest to give the
        # correct values. Tested with short arrays and confirmed this best matches the exact steps (maybe returns wrong
        # value when asking for value between DAC steps)
        interper = interp1d(fake_x, true_x, kind='nearest', bounds_error=False,
                            fill_value='extrapolate')
        return interper(x)


def i_sense_square_heated(x, mid, theta, amp, lin, const, hv, cc, hf, dS):
    """ Full transition signal with square wave heating and entropy change

    Args:
        x (Union[float, np.ndarray]):
        mid (Union[float, np.ndarray]):
        theta (Union[float, np.ndarray]):
        amp (Union[float, np.ndarray]):
        lin (Union[float, np.ndarray]):
        const (Union[float, np.ndarray]):
        hv (Union[float, np.ndarray]): Heating Voltage
        cc (Union[float, np.ndarray]): Cross Capacitance of HV and Plunger gate (i.e. shift middle)
        hf (Union[float, np.ndarray]): Heat Factor (i.e. how much HV increases theta)
        dS (Union[float, np.ndarray]): Change in entropy between N -> N+1

    Returns:
        (Union[float, np.ndarray]): evaluated function at x value(s)
    """
    heat = hf * hv ** 2  # Heating proportional to hv^2
    T = theta + heat  # theta is base temp theta, so T is that plus any heating
    X = x - mid + dS * heat - hv * cc  # X is x position shifted by entropy when heated and cross capacitance of heating
    arg = X / (2 * T)
    return -amp / 2 * np.tanh(arg) + lin * (x - mid) + const  # Linear term is direct interaction with CS
# endregion


# region Processing from I_sense to Entropy
"""All the functions for processing I_sense data into the various steps of square wave heated data"""


def chunk_data(data, awg):
    """
    Breaks up data into chunks which make more sense for square wave heating datasets.
    Args:
        data (np.ndarray): 1D or 2D data (full data to match original x_array).
            Note: will return with y dim regardless of 1D or 2D
        awg (AWG.AWG): AWG part of dat which has all square wave info in.

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
        start_index (Union[int, None]): Start index to average in each setpoint chunk
        fin_index (Union[int, None]): Final index to average to in each setpoint chunk (can be negative)

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
        start_cycle (Union[int, None]): Cycle to start averaging from
        fin_cycle (Union[int, None]): Cycle to finish averaging on (can be negative to count backwards)

    Returns:
        np.ndarray: Averaged data with shape ([ylen], setpoints, num_steps)

    """
    # [y], setpoints, numsteps, cycles
    data = np.array(binned_data, ndmin=4)  # [y], setpoints, numsteps, cycles
    averaged = np.mean(np.moveaxis(data, -1, 0)[start_cycle:fin_cycle], axis=0)
    if averaged.shape[0] == 1:  # Return 1D or 2D depending on y_len
        averaged = np.squeeze(averaged, axis=0)
    return averaged


def average_2D(x, data):
    """
    Averages data in y direction after centering using fits to v0 parts of square wave. Returns 1D data unchanged
    Args:
        x (np.ndarray): Original x_array for data
        data (np.ndarray): Data after binning and cycle averaging. Shape ([ylen], setpoints, num_steps)

    Returns:
        Tuple[np.ndarray, np.ndarray]: New x_array, averaged_data (shape (setpoints, num_steps))
    """
    if data.ndim == 3:
        z0s = data[:, (0, 2)]
        z0_avg_per_row = np.mean(z0s, axis=1)
        fits = T.transition_fits(x, z0_avg_per_row)
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
    return np.nancumsum(data, axis=-1)*sf


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


