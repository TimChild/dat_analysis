from src.CoreUtil import FIR_filter
from src.Scripts.StandardImports import *

from src.DatObject.Attributes import Entropy as E

import src.HDF_Util as HDU


def ntuple_to_df(ntuple: NamedTuple):
    d = ntuple._asdict()
    cols = list(d.keys())
    vals = list(d.values())
    df = pd.DataFrame(data=[vals], columns=cols)
    return df


def dict_to_df(d: dict):
    if any([not hasattr(v, '__iter__') for v in d.values()]):
        d = {k: [v] for k, v in d.items()}
    return pd.DataFrame.from_dict(d)


def to_df(data, datnum=None):
    if HDU._isnamedtupleinstance(data):
        df = ntuple_to_df(data)
    elif type(data) == dict:
        df = dict_to_df(data)
    else:
        raise NotImplementedError

    if datnum:
        df['datnum'] = datnum
    return df

def _plot_square_row_0(dats, axs = None):
    if axs is None:
        fig, axs = PF.make_axes(len(dats))
    else:
        assert len(axs) >= len(dats)
        fig = axs[0].figure

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

    fig.suptitle(f'Square wave entropy. Row 0 only')
    for ax in axs:
        ax.cla()

    for dat, x, harm, fit, ax in zip(dats, xs, harms, fits, axs):
        ax.plot(x, harm)
        ax.plot(x, fit.best_fit, label=f'dS={fit.best_values["dS"]:.3f}')
        info = dat.AWG.info
        PF.ax_setup(ax, f'Dat{dat.datnum}: Freq={info.measureFreq/info.wave_len:.1f}Hz, Cycles={info.num_cycles}',
                    dat.Logs.x_label, 'Entropy Signal', legend=True)

    for dat in dats:
        dat.Other.save_code(inspect.getsource(_plot_square_row_0), 'simple_harm_2')

    return axs


def _plot_low_f_noise_comparsion(dats, axs=None):
    if axs is None:
        fig, axs = PF.make_axes(4)
    else:
        assert len(axs) >= len(dats)
    for ax in axs:
        ax.cla()

    for dat, ax, t in zip(dats, axs[0:2], ['with', 'without']):
        x = dat.Data.x_array
        data = dat.Data.i_sense[0]
        ax.plot(x, data, label='Data', linewidth=1)
        filt = FIR_filter(data, dat.Logs.Fastdac.measure_freq, 5)
        ax.plot(x, filt, label='Cutoff = 5Hz', linewidth=2)
        PF.ax_setup(ax, f'Dat{dat.datnum}: I_sense {t} Low F noise\nMeasure_Freq={dat.Logs.Fastdac.measure_freq:.1f}/s',
                    dat.Logs.x_label, 'Current /nA', legend=True)

        axs[2].plot(x, filt, label=f'dat{dat.datnum}')
        PF.ax_setup(axs[2], f'Comparing filtered data directly',
                    dat.Logs.x_label, 'Current /nA', legend=True)

        PF.Plots.power_spectrum(data, dat.Logs.Fastdac.measure_freq, ax=axs[3], label=f'dat{dat.datnum}')

    for dat in dats:
        dat.Other.save_code(inspect.getsource(_plot_low_f_noise_comparsion), 'low_f_noise_comparison')


def _plot_entropy_comparison(dats):
    fig, ax = plt.subplots(1)

    ax.cla()

    srs_df = pd.DataFrame()
    fdac_df = pd.DataFrame()
    bdac_df = pd.DataFrame()
    for dat in dats:
        x = dat.Data.x_array - dat.Transition.avg_fit.best_values.mid
        data = dat.Data.enty
        if data.ndim == 2:
            data = data[0]

        PF.display_1d(x, data, ax, label=f'Dat{dat.datnum}')
        sdf = to_df(dat.Logs.srs1, dat.datnum)
        fdf = to_df(dat.Logs.fds, dat.datnum)
        bdf = to_df(dat.Logs.bds, dat.datnum)
        srs_df = srs_df.append(sdf)
        fdac_df = fdac_df.append(fdf)
        bdac_df = bdac_df.append(bdf)

    PF.ax_setup(ax, 'Comparing Entropy signal strength:\nSingle row of Entropy_y', dats[0].Logs.x_label, 'Current /nA',
                legend=True)

    for df in [srs_df, fdac_df, bdac_df]:
        df.reset_index(drop=True, inplace=True)
        df = df[['datnum', *list(set(df.columns.values) - {'datnum'})]]
        PF.plot_df_table(df, f'Comparing dats{df.datnum[df.first_valid_index()]}-{df.datnum[df.last_valid_index()]}',
                         sig_fig=3)

    for dat in dats:
        dat.Other.save_code(inspect.getsource(_plot_entropy_comparison), 'entropy_comparison')


if __name__ == '__main__':
    dats = get_dats(range(500, 515))
    _plot_square_row_0(dats)

    dats = get_dats([376, 486, 487])
    _plot_entropy_comparison(dats)

