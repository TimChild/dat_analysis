from src.Scripts.StandardImports import *



def load():
    C.load_dats()

def plot_avg_vs_time(dats, ax:plt.Axes):
    for dat in dats:
        dsavg = dat.Entropy.dS
        ds = dat.Entropy.fit_values.dSs
        # x = dat.Logs.time_completed
        x = [dat.datnum]*len(dat.Data.y_array)
        ax.scatter(x, ds, s=1)
        ax.scatter(x[0], dsavg, s=10, c='k')
    return ax

def set_labels(ax, sr):
    ax.set_title(f'Sweeprate = {sr:.0f}/mV/s')
    ax.set_ylabel('Entropy/Kb')
    ax.axhline(np.log(2), c="k", ls=":")
    ax.set_ylim(0.3, 1.2)

    return ax


if __name__ == '__main__':
    fig, ax = PF.reuse_plots(7)
    for i, datstart in enumerate((414,415,416,417,418,419,420)):
        dats = [make_dat_standard(datnum, dfoption='overwrite') for datnum in range(datstart, 446, 9)]
        for dat in dats:
            for p in dat.Entropy.params:
                p['const'].vary = True
            dat.Entropy.recalculate_fits(dat.Entropy.params)
        ax[i] = plot_avg_vs_time(dats, ax[i])
        set_labels(ax[i], dats[0].Logs.sweeprate)
    # fig.suptitle('Entropy vs Time')
    plt.tight_layout()
    