import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from scipy.interpolate import interp1d

from FinalFigures.Gamma.plots import dndt_2d


if __name__ == '__main__':
    from src.DatObject.Make_Dat import get_dats

    fit_name = 'forced_theta_linear'
    all_dats = get_dats(range(2095, 2125 + 1, 2))  # Goes up to 2141 but the last few aren't great
    tonly_dats = get_dats(range(2096, 2126 + 1, 2))
    # Loading fitting done in Analysis.Feb2021.entropy_gamma_final

    gamma_cg_vals = [dat.Logs.fds['ESC'] for dat in tonly_dats]
    gammas_over_thetas = [dat.Transition.get_fit(name=fit_name).best_values.g /
                          dat.Transition.get_fit(name=fit_name).best_values.theta for dat in tonly_dats]

    full_x = np.linspace(-150, 150, 1000)
    data = []
    for dat in all_dats:
        out = dat.SquareEntropy.get_Outputs(name=fit_name)
        x = out.x
        dndt = out.average_entropy_signal
        interper = interp1d(x=x, y=dndt, bounds_error=False)
        data.append(interper(x=full_x))


    # Do Plotting
    # 2D dN/dT
    fig, ax = plt.subplots(1, 1)
    ax = dndt_2d(ax, x=full_x, y=gammas_over_thetas, data=data)
    plt.tight_layout()
    fig.show()
