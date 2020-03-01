from src.Scripts.StandardImports import *
import lmfit as lm
from src.DatCode.DCbias import DCbias
from src.DatCode.Transition import Transition
import src.DatCode.DCbias as DC
import src.DatCode.Entropy as E


def add_dcbias(dat):
    """Temporary function to fix my dat pkl objects which were missing relevant info"""
    comment_keys = [key.strip() for key in dat.Logs.comments.split(',')]
    if dat.Data.i_sense is None and 'cscurrent_2d' in dat.Data.data_keys:  # Need to update some old dats
        dat.Data.i_sense = dat.Data.cscurrent_2d
        print(f'Added i_sense data for dat{dat.datnum:d}')

    if 'dcbias' in comment_keys and getattr(dat, 'Transition', None) is None:  # All DCbias scans should also have Transition attr
        dat._reset_transition()
        dat.dattype.append('transition')
        print(f'Added Transition attr for dat{dat.datnum:d}')

    if 'dcbias' in comment_keys and '2drepeat' not in comment_keys:
        dat._reset_dcbias()
        dat.dattype.append('dcbias')
        print(f'Added DCbias attr for dat{dat.datnum:d}')

    datdf = DF.DatDF()
    datdf.update_dat(dat)


def plot_dc_bias(single_dat=None, datnum_list = None):
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
        temp = dat.Logs.temps['mc']*1000
        PF.add_to_fig_text(fig, f'Temp = {temp:.1f}mK')
        PF.add_to_fig_text(fig, f'Sweeprate = {dat.Logs.sweeprate:.1f}mV/s')
        PF.plot_dac_table(axs[3], dat)


    # plot_dc_bias(datnum_list=[720, 722, 724, 727, 729, 731, 734, 736, 738, 741, 743, 745, 748])

def plot_integrated_entropy(dat):
    fig, axs = PF.make_axes(5)
    pf = dat.Entropy.standard_plot_function()
    pf(dat, axs, plots = [1,2,3,4,10])
    PF.plot_dac_table(axs[5], dat)
    fig.suptitle(f'Dat{dat.datnum}')
    plt.tight_layout()
    PF.add_standard_fig_info(fig)
    PF.add_to_fig_text(fig, f'sweeprate={dat.Logs.sweeprate:.0f}mV/s, temp = {dat.Logs.temp:.0f}mK')
    # x = dat.Entropy.integrated_entropy_x_array
    # y = dat.Entropy.integrated_entropy
    # err = dat.Entropy.scaling_err
    # ax[0].fill_between(x, y * (1 - err), y * (1 + err), color='#AAAAAA')
    # PF.display_1d(x, y, ax[0], y_label='Entropy/kB', dat=dat)
    


if __name__ == '__main__':
    # dcbias_dats = [make_dat_standard(num, dfoption='load') for num in [277, 278, 313, 771]]
    dt = 0.20
    dterr = 0.05

    dats = [make_dat_standard(datnum, dfoption='load') for datnum in range(418, 421)]
    for dat in dats:
        if dat.Transition.version != '1.2':
            dat._reset_transition()
            print('updated transition, need to update/save df')
        if dat.Entropy.version != '2.1':
            dat._reset_entropy()
            print('updated entropy, need to update/save df')
        amp = dat.Transition.amp
        amperr = np.std(dat.Transition.fit_values.amps)
        dat.Entropy.init_integrated_entropy_average(dt/2, dterr, amp, amperr)

    for dat in dats:
        plot_integrated_entropy(dat)

    # fig, ax = PF.make_axes(4)
    #
    # dat = make_dat_standard(420, dfoption='load')

    # x = dat.Entropy.x_array
    # entr = dat.Entropy.entrav
    # ax[0].plot(x, entr)
    # entr = np.flip(entr)
    # np.nancumsum(entr)
    # cumsum = np.nancumsum(entr)
    # dx = np.abs(x[1] - x[0])
    # cumsum_dx = cumsum * dx
    # x = np.flip(x)
    # ax[1].plot(x, cumsum_dx)
    # ax[1].set_title('No scaling just integrated')
    # ax[0].set_title('Raw entropy data (no scaling)')
    # isense = dat.Data.i_sense[0]
    # isense = np.flip(isense)
    # ax[2].plot(x, isense)
    # ax[2].set_title('I_sense ')
    # scaling = dx / 0.1 / dat.Transition.amp
    # amp = dat.Transition.amp
    # norm = cumsum_dx * scaling
    # ax[3].plot(x, norm)
    # ax[3].set_title('Normalized with sf')
    # fig.suptitle(f'Dat{dat.datnum}')