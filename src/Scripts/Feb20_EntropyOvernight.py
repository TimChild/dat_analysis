from src.Scripts.StandardImports import *
from src.DatCode.Entropy import plot_standard_entropy



def plot_entropy(dat):
    fig, axs = PF.make_axes(4)
    plot_standard_entropy(dat, axs, plots=(1,2,3,4))
    PF.add_standard_fig_info(fig)
    fig.suptitle(f'Dat{dat.datnum}')
    PF.add_to_fig_text(fig, f'Sweeprate = {dat.Logs.sweeprate:.0f}mV/s')
    PF.add_to_fig_text(fig, f'ACbias = {dat.Instruments.srs1.out/1e3/50e6*1e9:.1f}/nA')
    PF.add_to_fig_text(fig, f'SRSfreq = {dat.Instruments.srs1.freq:.1f}Hz')
    PF.add_to_fig_text(fig, f'timeconst = {dat.Instruments.srs1.tc:.1f}ms')
    PF.add_to_fig_text(fig, f'Avg dS = {dat.Entropy.dS:.3f}')
# dats = [make_dat_standard(datnum, dfoption='load') for datnum in range(317,372+1)]
# dats1 = dats[::3]
# dats2 = dats[1::3]
# dats3 = dats[2::3]

if __name__ == '__main__':
    dats = [make_dat_standard(datnum, dfoption='overwrite') for datnum in range(413, 421)]
    for dat in dats:


        if dat.Logs.sweeprate is None:
            dat.Logs.calc_sweeprate()


        plot_entropy(dat)
        plt.draw()
    for dat in dats:
        dat.datname = 'free const'
        datdf.update_dat(dat)
    datdf.save()
