from src.Scripts.StandardImports import *
import lmfit as lm
from src.DatCode.DCbias import DCbias
from src.DatCode.Transition import Transition
import src.DatCode.DCbias as DC
import src.DatCode.Transition as T
import src.DatCode.Entropy as E


def add_dcbias(dat):
    """Temporary function to fix my dat pkl objects which were missing relevant info"""
    comment_keys = [key.strip() for key in dat.Logs.comments.split(',')]
    if dat.Data.i_sense is None and 'cscurrent_2d' in dat.Data.data_keys:  # Need to update some old dats
        dat.Data.i_sense = dat.Data.cscurrent_2d
        print(f'Added i_sense data for dat{dat.datnum:d}')

    if 'dcbias' in comment_keys and getattr(dat, 'Transition',
                                            None) is None:  # All DCbias scans should also have Transition attr
        dat._reset_transition()
        dat.dattype.append('transition')
        print(f'Added Transition attr for dat{dat.datnum:d}')

    if 'dcbias' in comment_keys and '2drepeat' not in comment_keys:
        dat._reset_dcbias()
        dat.dattype.append('dcbias')
        print(f'Added DCbias attr for dat{dat.datnum:d}')

    datdf = DF.DatDF()
    datdf.update_dat(dat)


def plot_dc_bias(single_dat=None, datnum_list=None):
    """Plots DCbias info using DCbias plots and then adds some extra info in figtext and adds DAC values"""
    if datnum_list is not None:
        dats = [make_dat_standard(num, dfoption='load') for num in datnum_list]
    else:
        dats = [single_dat]
    for dat in dats:
        fig, axs = PF.make_axes(4)
        axs = DC.plot_standard_dcbias(dat, axs, plots=[1, 2, 3])
        fig.suptitle(f'Dat{dat.datnum}')
        PF.add_standard_fig_info(fig)
        temp = dat.Logs.temps['mc'] * 1000
        PF.add_to_fig_text(fig, f'Temp = {temp:.1f}mK')
        PF.add_to_fig_text(fig, f'Sweeprate = {dat.Logs.sweeprate:.1f}mV/s')
        PF.plot_dac_table(axs[3], dat)

    # plot_dc_bias(datnum_list=[720, 722, 724, 727, 729, 731, 734, 736, 738, 741, 743, 745, 748])


def plot_integrated_entropy(dat):
    fig, axs = PF.make_axes(5)
    pf = dat.Entropy.standard_plot_function()
    pf(dat, axs, plots=[1, 2, 3, 4, 10])
    PF.plot_dac_table(axs[5], dat)
    fig.suptitle(f'Dat{dat.datnum}')
    plt.tight_layout()
    PF.add_standard_fig_info(fig)
    PF.add_to_fig_text(fig, f'Gamma = {dat.Transition.g:.4f}mV')
    # x = dat.Entropy.integrated_entropy_x_array
    # y = dat.Entropy.integrated_entropy
    # err = dat.Entropy.scaling_err
    # ax[0].fill_between(x, y * (1 - err), y * (1 + err), color='#AAAAAA')
    # PF.display_1d(x, y, ax[0], y_label='Entropy/kB', dat=dat)

def reset_dats(dats):
    """Gets transition and Integration up to date. Also sets to di_gamma fitting for transition"""
    for dat in dats:
        if dat.Transition.version != T.Transition._Transition__version:
            dat._reset_transition()
            print(f'dat{dat.datnum}: Transition reset')
            datdf.update_dat(dat)
        if dat.Transition.fit_func != T.i_sense_digamma:
            print(f'dat{dat.datnum} changed to i_sense_digamma fit')
            dat.Transition.recalculate_fits(func=T.i_sense_digamma)
            datdf.update_dat(dat)
        if dat.Transition.g is None:
            print(f'dat{dat.datnum} reset avg transition values')
            dat.Transition.set_average_fit_values()
            datdf.update_dat(dat)
    # datdf.save()
    for dat in dats:
        print(f'dat{dat.datnum}: amp = {dat.Transition.amp:.4f}nA, g = {dat.Transition.g:3f}mV')
    for dat in dats:
        if hasattr(dat, 'Entropy'):
            if dat.Entropy._amp != dat.Transition.amp:
                print(f'dat{dat.datnum} int_entropy initialized')
                dat.Entropy.init_integrated_entropy_average(dt/2, dterr, dat.Transition.amp, np.std(dat.Transition.fit_values.amps))
                datdf.update_dat(dat)
            print(f'dat{dat.datnum}: Integrated_Entropy = {dat.Entropy.int_ds:.3f}/kB')
        # plot_integrated_entropy(dat)


def plot_entropy_vs_gamma():
    """Plots entropy vs gamma and amplitude vs gamma"""
    fig, axs = PF.make_axes(3)
    cs = plt.colormaps()
    gs = []
    nik_dss = []
    int_dss = []
    amps = []
    ampserr = []
    for dat in dats:
        g = dat.Transition.g
        nik_ds = dat.Entropy.dS
        int_ds = dat.Entropy.int_ds
        gs.append(g)
        nik_dss.append(nik_ds)
        int_dss.append(int_ds)
        amps.append(dat.Transition.amp)
        ampserr.append(np.std(dat.Transition.fit_values.amps))
    PF.display_1d(gs, nik_dss, axs[0], x_label='Gamma /mV', y_label='Entropy /kB', cmap='cool', marker='o',
                  label='Fitted dS')
    PF.display_1d(gs, int_dss, axs[0], x_label='Gamma /mV', y_label='Entropy /kB', cmap='cool', marker='x',
                  label='Integrated dS')
    axs[0].legend()

    PF.display_1d(gs, amps, axs[1], x_label='Gamma /mV', y_label='Amplitude /nA', errors=ampserr,
                  title='Di_gamma amplitudes')

    PF.plot_dac_table(axs[2], dats[0])

    PF.add_standard_fig_info(fig)
    PF.add_to_fig_text(fig, '')



def plot_transition_fits(dats):
    fig, axs = PF.make_axes(len(dats)*2)

    row=0

    for i, dat in enumerate(dats):
        ax1 = axs[2*i]
        ax2 = axs[2*i + 1]
        x = dat.Data.x_array
        data1d = dat.Transition._data[row]
        PF.display_2d(x, dat.Data.y_array, dat.Transition._data, ax1, dat=dat)
        PF.display_1d(x, data1d, ax2, dat=dat, scatter=True, label='i_sense data')
        ax2.plot(x, dat.Transition._full_fits[row].best_fit, label='di_gamma_fit', c='r')





if __name__ == '__main__':
    # dcbias_dats = [make_dat_standard(num, dfoption='load') for num in [277, 278, 313, 771]]
    dt = 0.20
    dterr = 0.02
    cfg.yes_to_all = False
    dats = [make_dat_standard(num, dfoption='load') for num in range(821, 838+1)]
    dats2 = dats[::4]


    # plot_transition_fits(dats2)
    # dats[0].plot_standard_info('qt', fit_attrs={'Transition': ['gs', 'amps', 'thetas', 'lins', 'consts']})