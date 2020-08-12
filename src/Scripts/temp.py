from src.Scripts.StandardImports import *

from scipy.interpolate import interp1d

from src.Plotting.Plotly.PlotlyUtil import PlotlyViewer
import plotly.graph_objects as go
import plotly.io as pio

from src.DatObject.Attributes import SquareEntropy as SE


def _plot_stability_vs_time(dats, ax1, ax2=None, avg_ent=False):
    """Plots transition stability on ax1, entropy stability on ax2"""
    fig = ax1.figure
    for ax in fig.axes:
        if not any([ax == a for a in [ax1, ax2]]):
            ax.remove()
        else:
            ax.cla()

    xts = []
    all_mids = []
    all_thetas = []
    all_dss = []
    for dat in dats:
        xt = np.linspace(0, dat.Logs.time_elapsed/3600, dat.Data.y_array.shape[0])  # A time axis in s for repeats
        mids = [fit.best_values.mid for fit in dat.Transition.all_fits]
        thetas = [fit.best_values.theta for fit in dat.Transition.all_fits]
        xts.append(xt)
        all_mids.append(mids)
        all_thetas.append(thetas)
        if dat.Entropy is not None:
            dss = [fit.best_values.dS for fit in dat.Entropy.all_fits]
            all_dss.append(dss)
        elif dat.SquareEntropy is not None:
            dss = [dat.SquareEntropy.Processed.outputs.entropy_fit.best_values.dS]
            all_dss.append(dss)

    ax = ax1
    ax.cla()
    axx: plt.Axes = ax.twiny()
    tt = 0
    top_labels = {}
    for xt, mids, thetas, dat in zip(xts, all_mids, all_thetas, dats):
        xt = xt + tt  # Add previous total time
        top_labels[np.mean(xt)] = dat.datnum
        ax.plot(xt, mids, label=f'{dat.datnum}')
        tt = xt[-1]+20/3600  # +20s roughly between scans

    # ax.legend(title='Datnum')
    PU.ax_setup(ax, 'Transition Center vs Time', 'Time /hours', 'Center /mV')

    top_labels = {k: top_labels[k] for k in sorted(top_labels.keys())}  # Make sure in order
    axx.set_xlim(ax.get_xlim())
    axx.set_xticks(np.array([k for k in top_labels.keys()])[::1])  # make tick for every dat
    axx.set_xticklabels(np.array([v for v in top_labels.values()])[::1], rotation='vertical')  # label each tick

    if ax2:
        ax = ax2
        ax.cla()
        axx: plt.Axes = ax.twiny()
        tt = 0
        top_labels = {}
        for xt, dss, dat in zip(xts, all_dss, dats):
            xt = xt + tt  # Add previous total time
            top_labels[np.mean(xt)] = dat.datnum
            if avg_ent:
                ax.scatter(np.mean(xt), np.mean(dss), label=f'{dat.datnum}')
            else:
                ax.plot(xt, dss, label=f'{dat.datnum}')
            tt = xt[-1] + 20 / 3600  # +20s roughly between scans

        # ax.legend(title='Datnum')
        PU.ax_setup(ax, 'Entropy vs Time', 'Time /hours', 'Entropy /kB')

        top_labels = {k: top_labels[k] for k in sorted(top_labels.keys())}  # Make sure in order
        axx.set_xlim(ax.get_xlim())
        axx.set_xticks(np.array([k for k in top_labels.keys()])[::1])  # make tick for every dat
        axx.set_xticklabels(np.array([v for v in top_labels.values()])[::1], rotation='vertical')  # label each tick

    for dat in dats:
        code = inspect.getsource(_plot_stability_vs_time)
        dat.Other.save_code(code, 'stability_vs_time')



if __name__ == '__main__':
    overwrite = False
    dats = get_dats(range(91, 117), overwrite=overwrite)

    show_plots = SE.ShowPlots(info=True, raw=False, setpoint_averaged=False, averaged=True, entropy=True)
    if overwrite is True:
        for dat in dats:
            dat.SquareEntropy.Processed.plot_info.show = show_plots
            dat.SquareEntropy.update_HDF()

    # dat = dats[0]
    # SE.plot_square_entropy(dat.SquareEntropy.Processed)

    x = [pd.to_datetime(dat.Logs.time_completed) for dat in dats]
    z = [dat.SquareEntropy.Processed.outputs.entropy_fit.best_values.dS for dat in dats]


    fig, axs = plt.subplots(2, 1)
    
