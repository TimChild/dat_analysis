from src.Scripts.StandardImports import *




def plot_integrated_entropy(dat):
    fig, axs = PF.make_axes(5)
    pf = dat.Entropy.standard_plot_function()
    pf(dat, axs, plots = [1,2,3,4])
    PF.plot_dac_table(axs[5], dat)
    fig.suptitle(f'Dat{dat.datnum}')
    plt.tight_layout()
    PF.add_standard_fig_info(fig)
    PF.add_to_fig_text(fig, f'ACbias = {dat.Instruments.srs1.out/50*np.sqrt(2):.1f}nA, sweeprate={dat.Logs.sweeprate:.0f}mV/s, temp = {dat.Logs.temp:.0f}mK')
    # x = dat.Entropy.integrated_entropy_x_array
    # y = dat.Entropy.integrated_entropy
    # err = dat.Entropy.scaling_err
    # ax[0].fill_between(x, y * (1 - err), y * (1 + err), color='#AAAAAA')
    # PF.display_1d(x, y, ax[0], y_label='Entropy/kB', dat=dat)



def plot_entropy_vs_acbias(dats):
    fig, ax = PF.make_axes(2)
    for dat in dats:
        if hasattr(dat, 'Entropy') is False:
            print(f'dat{dat.datnum} has no entropy')
            continue
        # dat = make_dat_standard(num, dfoption='load')
        if dat.Entropy.dS < 0 or dat.Entropy.dS > 1:
            print(f'dat{dat.datnum} has entropy = {dat.Entropy.dS}')
            continue
        bias = dat.Instruments.srs1.out/50*np.sqrt(2)
        ax[0].scatter(bias, dat.Entropy.dS, label=f'Dat{dat.datnum}', s=2)

    PF._optional_plotting_args(ax[0], x_label='AC bias /nA', y_label='Entropy /kB', title='Nik Entropy vs ACbias')
    ax[0].axhline(np.log(2), c='k', ls=':', label='Ln(2)')
    ax[0].legend(fontsize=4)
    PF.plot_dac_table(ax[1], dats[0])
    PF.add_standard_fig_info(fig)
    PF.add_to_fig_text(fig, f'Sweeprate = {dats[0].Logs.sweeprate:.0f}mV/s')




datnums = list(range(801,818+1))

dats = [make_dat_standard(num, dfoption='load') for num in datnums]
# for dat in dats:
#     datdf.update_dat(dat)
# datdf.save()

if __name__ == '__main__':
    plot_entropy_vs_acbias(dats)
    for dat in dats:
        if dat.datnum in [801, 802, 803, 805, 807, 809, 810]:
            # if dat.Entropy.int_entropy_initialized is False:
            fig, axs = PF.make_axes(4)
            plot_integrated_entropy(dat)
