from src.Scripts.StandardImports import *


def _plot_temp_and_stability(dats, ax: plt.Axes):
    xts = []
    all_mids = []
    all_thetas = []
    for dat in dats:
        xt = np.linspace(0, dat.Logs.time_elapsed/3600, dat.Data.y_array.shape[0])  # A time axis in s for repeats
        mids = [fit.best_values.mid for fit in dat.Transition.all_fits]
        thetas = [fit.best_values.theta for fit in dat.Transition.all_fits]
        xts.append(xt)
        all_mids.append(mids)
        all_thetas.append(thetas)


    axx: plt.Axes = ax.twiny()
    tt = 0
    prev_fridge_temp = 0
    top_labels = {}
    for xt, mids, thetas, dat in zip(xts, all_mids, all_thetas, dats):
        xt = xt + tt  # Add previous total time
        temp = dat.Logs.temps.mc
        top_labels[np.mean(xt)] = dat.datnum
        if np.isclose(prev_fridge_temp, temp, atol=0.005):  # if within 5mK
            color = ax.lines[-1].get_color()
            ax.plot(xt, mids, color=color)
            # axs[1].plot(xt, thetas, color=color)
        else:
            ax.plot(xt, mids, label=f'{temp * 1000:.0f}')
            # axs[1].plot(xt, thetas, label=f'{temp*1000:.0f}')
        prev_fridge_temp = temp
        tt = xt[-1] + 20 / 3600  # +20s roughly between scans
        if dat.datnum == 159:
            tt += 4.5  # 4.5 hours between dat159 and when I started again with dat187

    ax.legend(title='Fridge Temp /mK')
    PF.ax_setup(ax, 'Transition Center vs Time (and Temp)', 'Time /hours', 'Center /mV')
    # PF.ax_setup(axs[1], 'Theta vs Time(and Temp)', 'Time /hours', 'Theta /mV')

    top_labels = {k: top_labels[k] for k in sorted(top_labels.keys())}  # Make sure in order
    axx.set_xlim(ax.get_xlim())
    axx.set_xticks(np.array([k for k in top_labels.keys()])[::3])  # make tick for every dat
    axx.set_xticklabels(np.array([v for v in top_labels.values()])[::3], rotation='vertical')  # label each tick


def _plot_stability_vs_time(dats, ax1, ax2):
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
    PF.ax_setup(ax, 'Transition Center vs Time', 'Time /hours', 'Center /mV')

    top_labels = {k: top_labels[k] for k in sorted(top_labels.keys())}  # Make sure in order
    axx.set_xlim(ax.get_xlim())
    axx.set_xticks(np.array([k for k in top_labels.keys()])[::1])  # make tick for every dat
    axx.set_xticklabels(np.array([v for v in top_labels.values()])[::1], rotation='vertical')  # label each tick

    ax = ax2
    ax.cla()
    axx: plt.Axes = ax.twiny()
    tt = 0
    top_labels = {}
    for xt, dss, dat in zip(xts, all_dss, dats):
        xt = xt + tt  # Add previous total time
        top_labels[np.mean(xt)] = dat.datnum
        ax.plot(xt, dss, label=f'{dat.datnum}')
        tt = xt[-1] + 20 / 3600  # +20s roughly between scans

    # ax.legend(title='Datnum')
    PF.ax_setup(ax, 'Entropy vs Time', 'Time /hours', 'Entropy /kB')

    top_labels = {k: top_labels[k] for k in sorted(top_labels.keys())}  # Make sure in order
    axx.set_xlim(ax.get_xlim())
    axx.set_xticks(np.array([k for k in top_labels.keys()])[::1])  # make tick for every dat
    axx.set_xticklabels(np.array([v for v in top_labels.values()])[::1], rotation='vertical')  # label each tick

if __name__ == '__main__':
    # desc = '100mk stability'
    #
    # if desc == 'temp and stability':
    #     datnums = set(range(117, 213+1))
    #     datnums = datnums - set(list(range(160, 186+1)))
    #     dats = [get_dat(num) for num in datnums]
    # elif desc == '100mk stability':
    #     # Note to self: Forcing 'transition' only because I marked a couple as 'entropy' but the data was terrible
    #     dats = [get_dat(datnum, dattypes={'transition'}) for datnum in list(range(223, 253+1))]
    # else:
    #     raise ValueError
    #
    # dats = CU.order_list(dats, sort_by=[dat.datnum for dat in dats])
    #
    #
    # fig, axs = PF.make_axes(9)
    #
    # for ax in fig.axes:
    #     ax.cla()
    #
    # # _plot_temp_and_stability(dats, ax)
    # # _plot_temp_and_stability(dats, ax)
    #
    # dat: DatHDF
    # for ax, dat in zip(axs, [dd[num] for num in range(153, 159+1)]):
    #     x = dat.Data.x_array
    #     y = dat.Data.y_array
    #     data = dat.Data.i_sense
    #
    #     PF.display_2d(x, y, data, ax)
    #     PF.ax_setup(ax, f'Dat{dat.datnum}', 'RP/0.16 mV', 'Repeats')


    dats = [get_dat(num) for num in range(294, 327+1)]

