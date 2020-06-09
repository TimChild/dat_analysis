from src.Scripts.StandardImports import *
import scipy


def load():
    C.load_dats()


def plot_avg_vs_time(dats, ax: plt.Axes):
    for dat in dats:
        dsavg = dat.Entropy.avg_fit_values.dSs[0]
        ds = dat.Entropy.fit_values.dSs
        # x = dat.Logs.time_completed
        x = [dat.datnum] * len(dat.Data.y_array)
        ax.scatter(x, ds, s=1)
        ax.scatter(x[0], dsavg, s=10, c='k')
    return ax


def reset_dats(datdf):
    for i, datstart in enumerate((471, 474, 477)):
        dats = [make_dat_standard(datnum, dfoption='overwrite', datdf=datdf) for datnum in range(datstart, 717, 10)]
        for dat in dats:
            DF.update_save(dat, update=True, save=False, datdf=datdf)
    datdf.save()


def _plot_entropy_repeats(dats_list):
    """

    @param dats_list: List of dats (e.g. [dats1, dats2, dats3])
    @type dats_list: List[List[D.Dat]]
    @return: None
    @rtype: None
    """

    fig, ax = PF.make_axes(3)
    for i, dats in enumerate(dats_list):
        ax[i].cla()
        ax[i] = plot_avg_vs_time(dats, ax[i])
        PF.ax_setup(ax[i], f'Sweeprate = {dats[0].Logs.sweeprate:.0f}/mV/s\nDats[{dats[0].datnum}:{dats[-1].datnum}:10]',
                    'Datnum', 'Entropy /kB')
        ax[i].set_ylim(0.25, 1.3)
        ax[i].axhline(np.log(2), c="k", ls=":")

    plt.tight_layout()
    PF.add_standard_fig_info(fig)


if __name__ == '__main__':
    datdf = DF.DatDF(dfname='May20')
    dats1, dats2, dats3 = [[C.DatHandler.get_dat(datnum, 'base', datdf=datdf) for datnum in range(datstart, 717, 10)]
                           for datstart in (471, 474, 477)]
    _plot_entropy_repeats([dats1, dats2, dats3])

    fig, axs = plt.subplots(1,2)
    fig2, axs2 = plt.subplots(1, 2)
    fig3, axs3 = plt.subplots(1, 2)
    axs = axs.flatten()
    rows1 = [0] #[0, 3, 14]
    rows2 = [0] #[4, 15, 13]
    rows3 = [0] #[1, 2, 3]
    # for ax, dats, rows in zip(axs, [dats1, dats2, dats3], [rows1, rows2, rows3]):
    for ax, ax2, ax3, dats, rows in zip(axs, axs2, axs3, [dats1, dats3], [rows1, rows3]):
        dat_index = 0
        per_row = True
        cfg.PF_num_points_per_row = 1000


        # ax.cla()
        PF.ax_setup(ax, f'Sweeprate = {dats[0].Logs.sweeprate:.0f}mV/s\nDats[{dats[0].datnum}:{dats[-1].datnum}:10]',
                    'Plunger /mV', 'Entropy Signal /nA')
        dat = dats[dat_index]
        x = dat.Data.x_array
        if per_row is True:
            # data = dat.Entropy.entr
            # data = dat.Data.ADC1_2d
            data = dat.Data.ADC0_2d[0]
            x, data = CU.sub_poly_from_data(x, data*10, dat.Transition._full_fits[0])
            cfg.PF_binning = True
            # x, data = PF.bin_for_plotting(x, data)
            noise = scipy.fft(data)
            noise = np.square(np.abs(noise))
            x2 = np.linspace(0, 667, len(data))
            x2, noise = CU.bin_data([x2, noise], 10)
            noise_sum = np.nancumsum(noise)
            ax2.plot(x2, noise, label='fft')
            ax3.plot(x2, noise_sum)
            # ax.plot(x, data, label=f'{dat.datnum}, {0}, {dat.Entropy.fit_values.dSs[0]:.2f}', linewidth=1, markersize=5, marker='+')
            for row in rows:
                pass
                # PF.display_1d(x, data[row], ax, marker=None, auto_bin=True, linewidth=1,
                #                label=f'{dat.datnum}, {row}, {dat.Entropy.fit_values.dSs[row]:.2f}', scatter=True)
                # x, data = PF.bin_for_plotting(x, data)
                # ax.scatter(x, data[row], label=f'{dat.datnum}, {row}, {dat.Entropy.fit_values.dSs[row]:.2f}', s=2)
                # color = ax.lines[-1].get_c()
                # color = PF.adjust_lightness(color, -0.3)  # Make darker
                # ax.plot(x, dat.Entropy._full_fits[row].best_fit, color=color, linewidth=1)
            ax.legend(fontsize=8).set_title('Dat, Row, dS', prop={'size': 8})
        else:
            data = dat.Entropy.entrav
            PF.display_1d(x, data, ax, marker=None, auto_bin=True, linewidth=1,
                          label=f'{dat.datnum}, {dat.Entropy.avg_fit_values.dSs[0]:.2f}')
            ax.plot(dat.Entropy.avg_x_array, dat.Entropy._avg_full_fit.best_fit, color='C3', label='Fit')
            ax.legend(fontsize=8).set_title('Dat, dS', prop={'size': 8})

        PF.ax_text(ax, f'Bin_size ~ {round(len(x)/1000)}', loc=(0.05, 0.8))
    PF.add_standard_fig_info(fig)



    info = []
    col_names = ['Sweep_rate /mV/s', 'dS', 'dSerr', 'Reduced_Chi']
    for dats in [dats1, dats2, dats3]:
        std = np.average([dat.Entropy._avg_full_fit.params['dS'].stderr for dat in dats])
        red_chi = np.average([dat.Entropy._avg_full_fit.redchi for dat in dats])
        dS = np.average([dat.Entropy.avg_fit_values.dSs[0] for dat in dats])
        sr = dats[0].Logs.sweeprate
        info.append([sr, dS, std, red_chi])
    df = pd.DataFrame(info, columns=col_names)
    PF.plot_df_table(df, f'Scan quality vs sweeprate', sig_fig=3)


