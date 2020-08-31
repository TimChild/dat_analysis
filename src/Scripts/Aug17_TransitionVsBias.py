from src.Scripts.StandardImports import *
from dataclasses import dataclass


@dataclass
class TransitionBiasRepeats:
    dats: List[DatHDF] = field(repr=False)
    biases: List[float] = None
    fig: plt.Figure = field(default=None, repr=False)
    axs: List[plt.Axes] = field(default=None, repr=False)
    ax_dict: Dict[int, plt.Axes] = field(default=None, repr=False)
    datas: List[np.ndarray] = field(default_factory=list, repr=False)
    xs: List[np.ndarray] = field(default_factory=list, repr=False)
    datas_sub_poly: List[np.ndarray] = field(default_factory=list, repr=False)
    fits_sub_poly: List[np.ndarray] = field(default_factory=list, repr=False)


def get_transition_vs_bias_repeats(dats: List[DatHDF]) -> TransitionBiasRepeats:
    TBR = TransitionBiasRepeats(dats)

    biases = set([round(abs(dat.Logs.fds['R2T(10M)'])) for dat in dats])
    TBR.biases = biases

    fig, axs = P.make_axes(len(biases))
    TBR.fig, TBR.axs = fig, axs

    for ax in axs:
        ax.cla()

    ax_dict = {b: ax for b, ax in zip(sorted(biases), axs)}
    TBR.ax_dict = ax_dict
    for dat in dats:
        data = CU.decimate(dat.Transition.avg_data, dat.Logs.Fastdac.measure_freq, 50)
        TBR.datas.append(data)

        x = np.linspace(dat.Data.x_array[0], dat.Data.x_array[-1], data.shape[-1])

        dat.Transition.avg_fit.recalculate_fit(x, data)
        _, dsub = CU.sub_poly_from_data(x, data, dat.Transition.avg_fit.fit_result)
        _, fsub = CU.sub_poly_from_data(x, dat.Transition.avg_fit.eval_fit(x), dat.Transition.avg_fit.fit_result)
        TBR.datas_sub_poly.append(dsub)
        TBR.fits_sub_poly.append(fsub)

        x = x - dat.Transition.avg_fit.best_values.mid
        TBR.xs.append(x)

        bias = dat.Logs.fds['R2T(10M)']
        ax = ax_dict[round(abs(bias))]
        ax.plot(x, dsub, label=f'{dat.datnum}')
        ax.plot(x, fsub, label=f'{dat.datnum}_fit')

        PU.ax_setup(ax, f'Bias={abs(bias)}mV', 'LP*200 /mV', 'Current /nA', True)

    return TBR

