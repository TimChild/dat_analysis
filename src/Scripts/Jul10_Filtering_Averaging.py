from src.Scripts.StandardImports import *

from src.DatObject.Attributes.Transition import i_sense, transition_fits
from src.DatObject.Attributes.Entropy import entropy_nik_shape, entropy_fits


from scipy.signal import resample_poly, decimate
import h5py


def get_sweeprate(measure_freq, x_array: Union[np.ndarray, h5py.Dataset]):
    dx = np.mean(np.diff(x_array))
    mf = measure_freq
    return mf * dx


def max_variation(orig_vals: list, new_vals: list) -> Tuple[float, int]:
    """Just returns the max variation of any value in list passed and the index of that value"""

    max_var = 0
    index = None
    for i, (o, n) in enumerate(zip(orig_vals, new_vals)):
        if not np.isclose(o, 0, atol=0.0001):
            var = abs((n - o) / o)
            if var > max_var:
                index = i
                max_var = var
    return max_var, index


class FitStuff(object):
    def __init__(self, measure_freq, which):
        self.mf = measure_freq
        # For both
        self.mid = -1000
        self.theta = 0.25

        # For transition only
        self.amp = 0.6
        self.lin = 0.015
        self.const = 8

        # For entropy only
        self.dS = np.log(2)
        self.dT = 0.04

        self.start_vals = None
        self.set_start_vals(which)

    def set_start_vals(self, which):
        if which == 'transition':
            self.start_vals = [self.mid, self.theta, self.amp, self.lin, self.const]
        elif which == 'entropy':
            self.start_vals = [self.mid, self.theta, 0, self.dS, self.dT]
        else:
            raise ValueError


class Data(object):
    def __init__(self, x, z, desc, measure_freq, ofit, fit_name, fitter):
        self.x = x
        self.z = z
        self.desc = desc
        self.measure_freq = measure_freq
        self.ofit = ofit
        self.fit_name = fit_name
        self.fitter = fitter

    @classmethod
    def from_dat(cls, dat, which='transition'):
        x = dat.Data.x_array[:]
        if which == 'transition':
            # data = dat.Transition.avg_data[:].astype(np.float32)
            data = dat.Transition.data[0].astype(np.float32)
            dat.Transition.avg_fit.recalculate_fit(x, data, auto_bin=False)
            ofit = dat.Transition.avg_fit.fit_result
            fitter = transition_fits
        elif which == 'entropy':
            # data = dat.Entropy.avg_data[:].astype(np.float32)
            data = dat.Entropy.data[0].astype(np.float32)
            dat.Entropy.avg_fit.recalculate_fit(x, data, auto_bin=False)
            ofit = dat.Entropy.avg_fit.fit_result
            fitter = entropy_fits
        else:
            raise ValueError('Choose transition or entropy')
        return cls(dat.Data.x_array[:], data, f'Dat{dat.datnum}', dat.Logs.Fastdac.measure_freq, ofit, which, fitter)

    @classmethod
    def from_model(cls, fit_stuff: FitStuff, which='transition'):
        x = np.linspace(fit_stuff.mid - 20 * fit_stuff.theta, fit_stuff.mid + 20 * fit_stuff.theta,
                        num=round(2 * 20 * fit_stuff.theta * fit_stuff.mf))
        if which == 'transition':
            data = i_sense(x, *fit_stuff.start_vals)
            fitter = transition_fits
        elif which == 'entropy':
            data = entropy_nik_shape(x, *fit_stuff.start_vals)
            fitter = entropy_fits
        else:
            raise ValueError('Choose transition or entropy')
        ofit = fitter(x, data, auto_bin=False)[0]
        assert max_variation(fit_stuff.start_vals, ofit.best_values.values())[0] < 0.001
        return cls(x, data, 'Model', fit_stuff.mf, ofit, which, fitter)


class FilterInfo(object):
    def __init__(self, filterer, fig_title, x_label, leg_title, var_xs):
        self.filterer = filterer
        self.fig_title = fig_title
        self.x_label = x_label
        self.leg_title = leg_title
        self.var_xs = var_xs

    @classmethod
    def get_FI(cls, var_xs, which='low pass', measure_freq=None, bin_size=None, n_taps=1001, re_freq=20):
        if which == 'low pass':
            if measure_freq:
                filterer = filter_data_generator(measure_freq=measure_freq, n_taps=1001)
            else:
                raise ValueError('Provide measure_freq for lowpass filter')
            fig_title = f'Fit parameters vs Cutoff frequency: n_taps={n_taps}'
            x_label = 'Cutoff /Hz'
            leg_title = 'Cutoff /Hz, Max_variation/%'
        elif which == 'bin':
            filterer = bin_data_func
            fig_title = 'Fit parameters vs bin size'
            x_label = 'bin_size'
            leg_title = 'Bin size, Max_variation/%'
        elif which == 'both':
            if measure_freq and bin_size:
                filterer = filter_then_bin_generator(measure_freq=measure_freq, bin_size=bin_size, n_taps=n_taps)
            else:
                raise ValueError('measure_freq and/or bin_size missing')
            fig_title = f'Fit Params vs (Cutoff freq then bin): n_taps={n_taps}, bin_size={bin_size}'
            x_label = 'Cutoff /Hz'
            leg_title = 'Cutoff /Hz, Max_variation/%'
        elif which == 'resample':
            assert measure_freq
            filterer = resample_data_generator(measure_freq=measure_freq)
            fig_title = f'Fit parameters vs Resampled data'
            x_label = 'Resampled freq /Hz'
            leg_title = 'Resample /Hz, Max_variation/%'
        elif which == 'decimate':
            assert measure_freq
            filterer = decimate_data_generator(measure_freq)
            fig_title = f'Fit parameters vs Decimated data'
            x_label = 'Decimated freq /Hz'
            leg_title = 'Decimated /Hz, Max_variation/%'
        elif which == 'filter_decimate':
            assert measure_freq
            filterer = filter_then_decimate_generator(measure_freq=measure_freq, re_freq=re_freq)
            fig_title = f'Fit parameters vs Filtered then Decimated data ({re_freq:.1f}Hz)'
            x_label = 'Cutoff /Hz'
            leg_title = 'Cutoff /Hz, Max_variation/%'
        elif which == 'my_decimate':
            assert measure_freq
            filterer = my_decimate_generator(measure_freq)
            fig_title = f'Fit parameters vs Decimated data using my decimater'
            x_label = 'Cutoff /Hz'
            leg_title = 'Cutoff /Hz, Max_variation/%'
        else:
            raise ValueError
        return cls(filterer, fig_title, x_label, leg_title, var_xs)


def filter_data_generator(measure_freq, n_taps=1001):
    def filter_func(x, z, cutoff):
        nz = CU.FIR_filter(z, measure_freq, cutoff, edge_nan=True, n_taps=n_taps)
        nz, nx = CU.remove_nans(nz, x)
        return nx, nz
    return filter_func


def bin_data_func(x, z, bin_size):
    nx, nz = CU.bin_data([x, z], bin_size)
    return nx, nz


def resample_data_generator(measure_freq):
    def resample_func(x, z, re_freq):
        down = round(measure_freq/re_freq)  # Factor to decrease by to achieve ~ resample_freq
        nz = resample_poly(z, 1, down, axis=0, padtype='line')  # defaults to Kaiser window with 5.0 (not sure what the 5 does)
        nx = np.linspace(x[0], x[-1], num=nz.shape[0])
        return nx, nz
    return resample_func


def decimate_data_generator(measure_freq):
    def decimate_func(x, z, re_freq):
        down = round(measure_freq/re_freq)
        ntaps = 201
        nz = decimate(z, down, n=ntaps, ftype='fir', axis=0, zero_phase=True)
        bad = round(ntaps/down)  # Bad data points from filter on non-zero data
        nz = nz[bad:-bad]
        nx = np.linspace(x[ntaps], x[-ntaps-1], num=nz.shape[0])
        return nx, nz
    return decimate_func


def filter_then_bin_generator(measure_freq, bin_size, n_taps=1001):
    def filterer(x, z, cutoff):
        filter = filter_data_generator(measure_freq, n_taps=n_taps)
        nx, nz = filter(x, z, cutoff)
        nnx, nnz = bin_data_func(nx, nz, bin_size)
        return nnx, nnz
    return filterer


def filter_then_decimate_generator(measure_freq, re_freq):
    def filterer(x, z, cutoff):
        filter = filter_data_generator(measure_freq, n_taps=n_taps)
        nx, nz = filter(x, z, cutoff)
        resampler = resample_data_generator(measure_freq)
        nnx, nnz = resampler(nx, nz, re_freq)
        return nnx, nnz
    return filterer


def my_decimate_generator(measure_freq):
    def decimater(x, z, re_freq):
        nz = CU.decimate(z, measure_freq, re_freq, return_freq=False)
        nx = np.linspace(x[0], x[-1], nz.shape[-1])
        nz, nx = CU.remove_nans(nz, nx, verbose=False)
        return nx, nz

    # def decimater(x, z, re_freq):
    #     down = round(measure_freq/re_freq)
    #     true_freq = measure_freq/down
    #     cutoff = true_freq/2
    #     ntaps = 5*down
    #     if ntaps > 2000:
    #         logger.warning(f'Reducing measure_freq={measure_freq:.1f}Hz to {true_freq:.1f}Hz requires ntaps={ntaps} '
    #                        f'in FIR filter, which is a lot. Using 2000 instead')
    #         ntaps = 2000  # Will get very slow if using too many
    #     elif ntaps < 21:
    #         ntaps = 21
    #     # if cutoff < 5: logger.warning(f'Trying to decimate to {true_freq:.1f}Hz so lowpass filter at {cutoff:.1f}Hz '
    #     # f'wont be very effective')
    #     nz = CU.FIR_filter(z, measure_freq, cutoff, edge_nan=True, n_taps=ntaps)
    #     nz, nx = CU.remove_nans(nz, x, verbose=False)
    #     nz = np.squeeze(np.atleast_2d(nz)[:, ::down])  # To work on 1D or 2D data
    #     nx = np.squeeze(np.atleast_2d(nx)[:, ::down])
    #     return nx, nz
    return decimater


def plot_power_spec_of_model(Data: Data):
    ax = PF.Plots.power_spectrum(Data.z, Data.measure_freq, auto_bin=False)
    ax.set_xlim(0, 30)  # Only look at lowest 30Hz for pure signal
    PF.ax_setup(ax, f'Power Spectrum for {Data.fit_name} fit')
    return ax


def apply_to_data(fi: FilterInfo, D: Data):
    ddict = {name: [] for name in D.ofit.best_values.keys()}
    new = {'fits': [], 'xs': [], 'zs': []}
    for i, var_x in enumerate(fi.var_xs):
        nx, nz = fi.filterer(D.x, D.z, var_x)
        nfit = D.fitter(nx, nz, params=D.ofit.params, auto_bin=False)[0]
        for k, v in nfit.best_values.items():
            ddict[k].append(v)
        new['fits'].append(nfit)
        new['xs'].append(nx)
        new['zs'].append(nz)
    return ddict, new


def plot_variations(ax, new, D, FI):
    max_var_so_far = 0
    for i, var_x in enumerate(FI.var_xs):
        nfit = new['fits'][i]
        nx = new['xs'][i]
        nz = new['zs'][i]
        max_change, idx = max_variation(D.ofit.best_values.values(), nfit.best_values.values())
        if max_change > max_var_so_far + 0.001 or i == len(FI.var_xs) - 1:
            max_var_so_far = max_change
            PF.display_1d(nx, nz, ax, label=f'{var_x:.0f}: {max_change * 100:.3f}', auto_bin=False, linewidth=1,
                          marker='')
    ax.legend(title=FI.leg_title)
    fig = ax.figure
    fig.suptitle(FI.fig_title)


def plot_fit_params(axes, data_dict, data_inst, fit_info_inst):
    for k, ax in zip(data_dict.keys(), axes):
        nvs = data_dict[k]
        ov = data_inst.ofit.best_values[k]
        if not np.isclose(ov, 0, atol=0.0001):
            variation = (np.array(nvs) - ov) / ov * 100
        else:
            variation = np.array(nvs)
        PF.display_1d(fit_info_inst.var_xs, variation, ax, x_label=fit_info_inst.x_label, y_label='change %', auto_bin=False)
        PF.ax_setup(ax, f'{k}')


run = 'all'
if __name__ == '__main__' and run == 'all':
    # Which to run
    use = 'dat'  # 'dat' or 'model'
    data_type = 'transition'  # 'transition' or 'entropy'
    filter_type = 'my_decimate'  # 'low pass', 'bin', 'both', 'resample', 'decimate', 'filter_decimate', 'my_decimate'
    # filter_vars = np.linspace(50, 1, 50)  # low pass cutoff freqs
    # filter_vars = [round(len(d.x)/x) for x in np.linspace(1000, 100, 9)]  # bin sizes
    filter_vars = np.linspace(95, 5, num=45)  # resample frequencies
    n_taps = 2001  # Only used for 'low pass' or 'both'
    bin_size = 50  # Only used for filter_type == 'both'. How much to bin_data after filtering
    re_freq = 20  # Only used for filter_type == 'filter_decimate'. What freq to aim for in decimate
    show_power_spec = False  # power spectrum of dat or model
    reuse_axs = False

    # Other params for choices
    if use == 'dat':
        dat = get_dat(487)  # Entropy at 100mK 502/s Measure freq, 1mV/s sweeprate, 6nA bias
    elif use == 'model':
        mf = 1004  # Measure frequency
    else:
        raise ValueError

    # setup choice
    if use == 'dat':
        d = Data.from_dat(dat, data_type)
    elif use == 'model':
        FS = FitStuff(mf, data_type)
        d = Data.from_model(FS, data_type)
    else:
        raise ValueError

    # Power spectrum of data being used
    if show_power_spec:
        ax = plot_power_spec_of_model(d)

    # Make axes for each fit parameter and data
    if not reuse_axs:
        fig, axs = PF.make_axes(len(d.ofit.best_values) + 1, single_fig_size=(4, 4))
    for ax in axs:
        ax.cla()

    # Plot original data
    PF.display_1d(d.x, d.z, axs[0], 'Plunger/0.16 /mV', 'Current /nA', auto_bin=False, label=f'{d.desc} data',
                  marker='', linewidth=2, color='k')
    PF.ax_setup(axs[0], f'{d.desc} data for {d.fit_name}', legend=True)

    # Loop through some various levels of filtering and append fit values to ddict
    if filter_type == 'low pass':
        FI = FilterInfo.get_FI(filter_vars, 'low pass', measure_freq=d.measure_freq, n_taps=n_taps)
    elif filter_type == 'bin':
        FI = FilterInfo.get_FI(filter_vars, 'bin')
    elif filter_type == 'both':
        FI = FilterInfo.get_FI(filter_vars, 'both', measure_freq=d.measure_freq, bin_size=bin_size, n_taps=n_taps)
    elif filter_type == 'resample':
        FI = FilterInfo.get_FI(filter_vars, which='resample', measure_freq=d.measure_freq)
    elif filter_type == 'decimate':
        FI = FilterInfo.get_FI(filter_vars, 'decimate', measure_freq=d.measure_freq)
    elif filter_type == 'filter_decimate':
        FI = FilterInfo.get_FI(filter_vars, 'filter_decimate', measure_freq=d.measure_freq)
    elif filter_type == 'my_decimate':
        FI = FilterInfo.get_FI(filter_vars, 'my_decimate', measure_freq=d.measure_freq)
    else:
        raise ValueError

    # Calculate new fit values and fit info (x, z, fit)
    ddict, new = apply_to_data(FI, d)

    # Add to first graph
    plot_variations(axs[0], new, d, FI)

    # Plot variation of fit params
    plot_fit_params(axs[1:], ddict, d, FI)

    fig = axs[0].figure
    fig.tight_layout(rect=[0, 0, 1, 0.95])

