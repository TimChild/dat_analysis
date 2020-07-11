from src.Scripts.StandardImports import *

from src.DatObject.Attributes.Transition import i_sense, transition_fits
from src.DatObject.Attributes.Entropy import entropy_nik_shape, entropy_fits

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
            data = dat.Transition.avg_data[:].astype(np.float32)
            dat.Transition.avg_fit.recalculate_fit(x, data, auto_bin=False)
            ofit = dat.Transition.avg_fit.fit_result
            fitter = transition_fits
        elif which == 'entropy':
            data = dat.Entropy.avg_data[:].astype(np.float32)
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
    def get_FI(cls, var_xs, which='low pass', measure_freq=None):
        if which == 'low pass':
            if measure_freq:
                filterer = filter_data_generator(measure_freq)
            else:
                raise ValueError('Provide measure_freq for lowpass filter')
            fig_title = 'Fit parameters vs Cutoff frequency'
            x_label = 'cutoff'
            leg_title = 'Cutoff /Hz, Max_variation/%'
        elif which == 'bin':
            filterer = bin_data_func
            fig_title = 'Fit parameters vs bin size'
            x_label = 'bin_size'
            leg_title = 'Bin size, Max_variation/%'
        else:
            raise ValueError
        return cls(filterer, fig_title, x_label, leg_title, var_xs)


def filter_data_generator(measure_freq):
    def filter_func(x, z, cutoff):
        nz = CU.FIR_filter(z, measure_freq, cutoff, edge_nan=True, n_taps=1001)
        nz, nx = CU.remove_nans(nz, x)
        return nx, nz

    return filter_func


def bin_data_func(x, z, bin_size):
    nx, nz = CU.bin_data([x, z], bin_size)

    return nx, nz


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
    for i, (var_x) in enumerate(zip(FI.var_xs)):
        nfit = new['fits'][i]
        nx = new['xs'][i]
        nz = new['zs'][i]
        max_change, idx = max_variation(D.ofit.best_values.values(), nfit.best_values.values())
        if max_change > max_var_so_far + 0.001 or i == len(var_xs) - 1:
            max_var_so_far = max_change
            PF.display_1d(nx, nz, ax, label=f'{var_x:.0f}: {max_change * 100:.3f}', auto_bin=False, linewidth=1,
                          marker='')
    ax.legend(title=FI.leg_title)
    fig.suptitle(FI.fig_title)


def plot_fit_params(axes, data_dict, data_inst, fit_info_inst):
    for k, ax in zip(data_dict.keys(), axes):
        nvs = data_dict[k]
        ov = data_inst.ofit.best_values[k]
        if not np.isclose(ov, 0, atol=0.0001):
            variation = (np.array(nvs) - ov) / ov * 100
        else:
            variation = np.array(nvs)
        PF.display_1d(var_xs, variation, ax, x_label=fit_info_inst.x_label, y_label='change %', auto_bin=False)
        PF.ax_setup(ax, f'{k}')


run = 'all'
if __name__ == '__main__' and run == 'all':
    # Which to run
    use = 'dat'
    data_type = 'transition'
    filter_type = 'low pass'
    filter_vars = np.linspace(50, 1, 50)
    # filter_vars = [round(len(d.x)/x) for x in np.linspace(1000, 100, 9)]
    show_power_spec = False

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
    fig, axs = PF.make_axes(len(d.ofit.best_values) + 1, single_fig_size=(4, 4))
    for ax in axs:
        ax.cla()

    # Plot original data
    PF.display_1d(d.x, d.z, axs[0], 'Plunger/0.16 /mV', 'Current /nA', auto_bin=False, label=f'{d.desc} data',
                  marker='', linewidth=2, color='k')
    PF.ax_setup(axs[0], f'{d.desc} data for {d.fit_name}', legend=True)

    # Loop through some various levels of filtering and append fit values to ddict
    if filter_type == 'low pass':
        FI = FilterInfo.get_FI(filter_vars, 'low pass')
    elif filter_type == 'bin':
        FI = FilterInfo.get_FI(filter_vars, 'bin')
    else:
        raise ValueError

    # Calculate new fit values and fit info (x, z, fit)
    ddict, new = apply_to_data(FI, d)

    # Add to first graph
    plot_variations(axs[0], new, d, FI)

    # Plot variation of fit params
    plot_fit_params(axs[1:], ddict, d, FI)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
