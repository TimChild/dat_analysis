import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

from FinalFigures.Gamma.plots import integrated_entropy, entropy_vs_coupling
from src.UsefulFunctions import save_to_igor_itx


if __name__ == '__main__':
    from src.DatObject.Make_Dat import get_dats, get_dat

    ##########################################################################
    # Data for integrated_entropy
    fit_name = 'forced_theta_linear'
    dats = get_dats(range(2164, 2170 + 1, 3)) + [get_dat(2216)]
    tonly_dats = get_dats([dat.datnum + 1 for dat in dats])

    outs = [dat.SquareEntropy.get_Outputs(name=fit_name) for dat in dats]
    int_infos = [dat.Entropy.get_integration_info(name=fit_name) for dat in dats]

    xs = [out.x for out in outs]
    int_entropies = [int_info.integrate(out.average_entropy_signal) for int_info, out in zip(int_infos, outs)]
    gts = [dat.Transition.get_fit(name=fit_name).best_values.g / dat.Transition.get_fit(name=fit_name).best_values.theta
           for dat in tonly_dats]

    save_to_igor_itx(file_path=f'fig3_integrated_entropy.itx',
                     xs=xs + [np.arange(len(gts))],
                     datas=int_entropies + [np.array(gts)],
                     names=[f'int_entropy_{i}' for i in range(len(int_entropies))] + ['gts_for_int_entropies'],
                     x_labels=['Sweep Gate /mV'] * len(int_entropies) + ['index'],
                     y_labels=['Entropy /kB']*len(int_entropies) + ['G/T'])

    # Plot Integrated Entropy
    fig, ax = plt.subplots(1, 1)
    integrated_entropy(ax, xs=xs, datas=int_entropies, gamma_over_ts=gts)
    plt.tight_layout()
    fig.show()

    ############################################################################
    # Data for entropy_vs_coupling
    fit_name = 'forced_theta_linear'
    dats = get_dats(range(2095, 2125 + 1, 2))  # Goes up to 2141 but the last few aren't great
    tonly_dats = get_dats(range(2096, 2126 + 1, 2))

    gamma_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in tonly_dats])
    gammas = np.array([dat.Transition.get_fit(name=fit_name).best_values.g for dat in tonly_dats])

    int_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in dats])
    # TODO: Need to make sure all these integrated entropies are being calculated at good poitns (i.e. not including slopes)
    integrated_entropies = np.array([np.nanmean(
        dat.Entropy.get_integrated_entropy(name=fit_name,
                                           data=dat.SquareEntropy.get_Outputs(
                                               name=fit_name, check_exists=True).average_entropy_signal
                                           )[-10:]) for dat in dats])

    fit_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in dats if dat.Logs.fds['ESC'] < -260])
    fit_entropies = np.array(
        [dat.Entropy.get_fit(name=fit_name).best_values.dS for dat in dats if dat.Logs.fds['ESC'] < -260])

    save_to_igor_itx(file_path=f'fig3_entropy_vs_gamma.itx', xs=[fit_cg_vals, int_cg_vals],
                     datas=[fit_entropies, integrated_entropies],
                     names=['fit_entropy_vs_coupling', 'integrated_entropy_vs_coupling'],
                     x_labels=['Coupling Gate /mV'] * 2,
                     y_labels=['Entropy /kB', 'Entropy /kB'])

    # Plot entropy_vs_coupling
    fig, ax = plt.subplots(1, 1)
    ax = entropy_vs_coupling(ax, int_coupling=int_cg_vals, int_entropy=integrated_entropies,
                             fit_coupling=fit_cg_vals, fit_entropy=fit_entropies)
    plt.tight_layout()
    fig.show()
