from src.Scripts.StandardImports import *
import copy



if __name__ == '__main__':
    od = get_dat(1672, ESI_class=JanESI)
    od2 = get_dat(2786, ESI_class=JanESI)
    nd = get_dat(220)

    fig3, ax3 = plt.subplots(1)
    fig2, ax2 = plt.subplots(1)
    fig, ax = plt.subplots(1)
    ax3.cla()
    ax2.cla()
    ax.cla()
    raw = True
    for dat, txt, rawkey in zip([nd, od, od2], ['New', 'Old 2008/s', 'Old 372/s'], ['Exp_cscurrent_2d_RAW', 'Exp_ADC0_2d', 'Exp_ADC0_2d']):
        row = 0
        label = f'{txt} dat{dat.datnum}'
        if raw is False:
            x = dat.Data.x_array
            z = dat.Data.i_sense[row]
            fv = dat.Transition.all_fits[row].best_values
            bf = dat.Transition.all_fits[row].eval_fit(x)  # power spec z
        else:
            x = dat.Data.x_array[:]
            z = dat.Data.__getattr__(rawkey)[row]
            if txt == 'New':  # Records in mV now...
                z = z/1000
            elif txt == 'Old 2008/s':  # Lower CS bias
                z = z*3
            fi = copy.deepcopy(dat.Transition.all_fits[row])
            fi.recalculate_fit(x, z, auto_bin=True)
            fv = fi.best_values
            bf = fi.eval_fit(x)

        psz = z - bf

        x = x - fv.mid
        z = z - x*fv.lin - fv.const
        z = z/fv.amp
        PF.display_1d(x, z, ax, dat.Logs.x_label, 'Current /nA', label=label, auto_bin=False)

        PF.Plots.power_spectrum(psz, dat.Logs.Fastdac.measure_freq, 1, ax2, label=label, linewidth=0.5, marker='')
        # PF.display_1d(x, psz, ax3, x_label=dat.Logs.x_label, label=label, linewidth=0.2)

    PF.ax_setup(ax, f'Comparing Noise from Jan to Jun: Raw={raw}', legend=True)
    PF.ax_setup(ax2, f'Comparing Noise from Jan to Jun: Raw={raw}', legend=True)


