import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

import src.UsefulFunctions as U

from FinalFigures.Gamma.plots import getting_amplitude_and_dt, dndt_signal


if __name__ == '__main__':
    from src.DatObject.Make_Dat import get_dats, get_dat

    #############################################################################################

    # Data for single hot/cold plot
    fit_name = 'forced_theta_linear'
    dat = get_dat(2164)
    out = dat.SquareEntropy.get_Outputs(name=fit_name)
    sweep_x = out.x
    cold_transition = np.nanmean(out.averaged[(0, 2), :], axis=0)
    hot_transition = np.nanmean(out.averaged[(1, 3), :], axis=0)

    U.save_to_igor_itx(file_path=f'fig1_hot_cold.itx', xs=[sweep_x] * 2, datas=[cold_transition, hot_transition],
                       names=['cold', 'hot'], x_labels=['Sweep Gate /mV'] * 2, y_labels=['Current /nA'] * 2)

    # Plotting for Single hot/cold plot
    fig, ax = plt.subplots(1, 1)
    getting_amplitude_and_dt(ax, x=sweep_x, cold=cold_transition, hot=hot_transition)
    plt.tight_layout()
    fig.show()

    # Data for dN/dT
    fit_name = 'forced_theta_linear'
    # dats = get_dats(range(2164, 2170 + 1, 3)) + [get_dat(2216)]
    all_dats = get_dats([2164, 2216])  # Weak, Strong coupling
    tonly_dats = get_dats([dat.datnum + 1 for dat in all_dats])

    outs = [dat.SquareEntropy.get_Outputs(name=fit_name) for dat in all_dats]
    int_infos = [dat.Entropy.get_integration_info(name=fit_name) for dat in all_dats]

    xs = [out.x for out in outs]
    dndts = [out.average_entropy_signal for out in outs]
    gts = [dat.Transition.get_fit(name=fit_name).best_values.g / dat.Transition.get_fit(name=fit_name).best_values.theta
           for dat in tonly_dats]

    U.save_to_igor_itx(file_path=f'fig1_dndt.itx', xs=xs + [np.arange(4)], datas=dndts + [np.array(gts)],
                       names=[f'dndt_{i}' for i in range(len(dndts))] + ['gts_for_dndts'],
                       x_labels=['Sweep Gate /mV'] * len(dndts) + ['index'],
                       y_labels=['dN/dT /nA'] * len(dndts) + ['G/T'])

    # dNdT Plots (one for weakly coupled only, one for strongly coupled only)
    fig, ax = plt.subplots(1, 1)
    ax = dndt_signal(ax, xs=xs[0], datas=dndts[0])
    ax.set_xlim(-100, 100)
    ax.set_title('dN/dT for weakly coupled')
    plt.tight_layout()
    fig.show()

    fig, ax = plt.subplots(1, 1)
    dndt_signal(ax, xs=xs[1], datas=dndts[1])
    ax.set_title('dN/dT for gamma broadened')
    plt.tight_layout()
    fig.show()
