from src.Scripts.StandardImports import *
from src.Scripts.Jun30_dot_tuning import _plot_dat

from src.DatObject.Attributes import Entropy as E

import inspect


if __name__ == '__main__':
    dats = get_dats(range(500, 515))

    fits = []
    xs = []
    harms = []
    for dat in dats:
        sp_len = dat.AWG.AWs[0][1][0]
        w0, wp, wm = dat.AWG.get_full_wave_masks(0)
        x = dat.Data.x_array
        z = dat.Data.i_sense[0]
        xb, zb, w0b, wpb, wmb = CU.bin_data([x, z, w0, wp, wm], sp_len)
        z0 = zb * w0b
        zp = zb * wpb
        zm = zb * wmb
        z0 = z0[~np.isnan(z0)]
        zp = zp[~np.isnan(zp)]
        zm = zm[~np.isnan(zm)]
        z0 = CU.bin_data(z0, 2)
        x = CU.bin_data(xb, 4)
        harm = (zm + zp) / 2 - z0
        harm = harm * -1
        fit = E.entropy_fits(x, harm)[0]

        harms.append(harm)
        fits.append(fit)
        xs.append(x)

    fig, axs = PF.make_axes(len(dats))

    fig.suptitle(f'Square wave entropy. Row 0 only')
    for ax in axs:
        ax.cla()

    for dat, x, harm, fit, ax in zip(dats, xs, harms, fits, axs):
        ax.plot(x, harm)
        ax.plot(x, fit.best_fit, label=f'dS={fit.best_values["dS"]:.3f}')
        info = dat.AWG.info
        PF.ax_setup(ax, f'Dat{dat.datnum}: Freq={info.measureFreq/info.wave_len:.1f}Hz, Cycles={info.num_cycles}', dat.Logs.x_label, 'Entropy Signal', legend=True)

