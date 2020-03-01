from src.Scripts.StandardImports import *


import src.Core as C

def plot_theta_vs_temp(dats):
    temps = [dat.Logs.temps['mc'] * 1000 for dat in dats]
    avg_thetas = [dat.Transition.theta for dat in dats]

    all_temps = [[dat.Logs.temps['mc'] * 1000] * len(dat.Data.y_array) for dat in dats]
    all_thetas = [dat.Transition.fit_values.thetas for dat in dats]

    fig, ax = plt.subplots(1)
    ax.plot(temps, avg_thetas)
    ax.scatter(all_temps, all_thetas)
    plt.show()
    PF._optional_plotting_args(ax, **{'x_label': 'Temperature /mK', 'y_label': 'Theta/mV (0.16 divider)'})
    plt.grid('on')
    PF.add_standard_fig_info(fig)
    return fig, ax

def get_dats(start, num, step, dattypes=None, dfoption='sync'):
    dats = list(range(start, start+num*step + 1, step))
    dats = [make_dat_standard(datnum, dfoption=dfoption, dattypes=dattypes) for datnum in dats]
    return dats

if __name__ == '__main__':
    PF.mpluse('qt')
    dat = make_dat_standard(281, dfoption='load')



