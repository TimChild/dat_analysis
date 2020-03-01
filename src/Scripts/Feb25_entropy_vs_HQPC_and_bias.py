from src.Scripts.StandardImports import *



datnums = list(range(719, 747+1))
plot_dats = [719, 721, 723, 726, 728, 730, 733, 735, 737]
calibration_dats = set(datnums)-set(plot_dats)

def dataset_summary(plotnum=4):
    HQPC_resistances = [20, 15, 13]
    SRSout_multiples = [0.8, 1, 1.2]
    
    fig, axs = PF.make_axes(9)
    for datnum, ax in zip(plot_dats, axs):
        dat = make_dat_standard(datnum, dfoption='load')
        assert 'entropy' in dat.dattype
        pf = dat.Entropy.standard_plot_function()
        pf(dat, [ax], plots=[plotnum], kwargs_list=[{'no_datnum': False}])  # 10
        ax.set_title(f'HQPC={dat.Logs.fdacs[0]:.0f}mV, ACbias={dat.Instruments.srs1.out/50*np.sqrt(2):.1f}nA')
    
    PF.add_standard_fig_info(fig)
    PF.add_to_fig_text(fig, 'HQPC resistance = 20K, 15K, 13K(plateau)')
    fig.suptitle('Nik Entropy vs HQPC and Varying Bias')

def entropy_per_dataset(dat):
    fig, axs = PF.make_axes(6)
    pf = dat.Entropy.standard_plot_function()
    pf(dat, axs, plots=(1,2,3,4,10))
    PF.plot_dac_table(axs[-1], dat)
    PF.add_standard_fig_info(fig)
    fig.suptitle(f'Dat{dat.datnum}: Entropy Summary')
    
if __name__ == '__main__':
    dats = [make_dat_standard(num, dfoption='load') for num in plot_dats[3:6]]
    for dat in dats:
        if dat.Entropy.int_entropy_initialized is False:
            dcbias_dat = make_dat_standard(dat.datnum + 1, dfoption='load')
            dT = dcbias_dat.DCbias.get_dt_at_current(dat.Instruments.srs1.out / 50 * np.sqrt(2))
            pf = dcbias_dat.DCbias.standard_plot_function()
            fig, axs = PF.make_axes(4)
            pf(dcbias_dat, axs, plots=[1,2,3,4])
            dat.Entropy.init_integrated_entropy_average(dT_mV=dT/2, dT_err=0, amplitude=dat.Transition.amp, amplitude_err=np.std(dat.Transition.fit_values.amps))
        entropy_per_dataset(dat)