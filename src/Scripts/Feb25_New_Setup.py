from src.Scripts.StandardImports import *
from src.DatCode.DCbias import plot_standard_dcbias
import src.DatCode.DCbias as DC
from src.DatCode.Entropy import plot_standard_entropy
from typing import List
from src.DatCode import Transition as T
import src.DatCode.Entropy as E
import src.DatCode.Dats as Dats

def plot_dc_bias(dat):
    fig, axs = PF.make_axes(4)
    plot_standard_dcbias(dat, axs, plots=[1,2,3,4])

def plot_integrated(dat):
    fig, ax = PF.make_axes(6)
    plot_standard_entropy(dat, ax, plots=[1, 2, 3, 4, 10])
    PF.plot_dac_table(ax[-1], dat)
    fig.suptitle(f'Dat{dat.datnum}')
    PF.add_standard_fig_info(fig)
    PF.add_to_fig_text(fig, f'dT = {dat.Entropy._int_dt:.3f}mV')
    PF.add_to_fig_text(fig, f'amp = {dat.Transition.amp:.3f}nA')
    plt.tight_layout(rect=(0.0, 0.10, 1, 0.95), pad=0.5)


def set_integrated_entropy(dat, dcbiasdat, save=True):
    y2a = cfg.yes_to_all
    cfg.yes_to_all = True
    datdf = DF.DatDF()
    dT = dcbiasdat.DCbias.get_dt_at_current(dat.Instruments.srs1.out/50*np.sqrt(2))
    dat.Entropy.init_integrated_entropy_average(dT/2, dT_err=0.2*dT/2, amplitude=dat.Transition.amp, amplitude_err=np.std(dat.Transition.fit_values.amps))
    datdf.update_dat(dat)
    if save is True:
        datdf.save()
    cfg.yes_to_all = y2a



def plot_nik_fit(dats):
    fig, axs = PF.make_axes(len(dats)*2)
    
    for i, dat in enumerate(dats):
        ax1 = axs[2*i]
        ax2 = axs[2*i+1]
        
        x = dat.Entropy.x_array
        y = dat.Entropy.entrav
        fit = dat.Entropy._full_fit_average
        
        PF.display_2d(x, dat.Data.y_array, dat.Entropy._data, ax1, dat=dat)
        PF.display_1d(x, y, ax=ax2, scatter=True, dat=dat, label='Averaged Entr')
        ax2.plot(x, fit.best_fit, label='Best fit', c='r')
        ax2.legend()
        ds = fit.best_values['dS']
        dt = fit.best_values['dT']
        theta = fit.best_values['theta']
        PF.ax_text(ax2, f'dS={ds:.3f}\ndT={dt:.3f}nA\ntheta={theta:.3f}mV')
    fig.suptitle('Nik Fits')

"""
    DCbiasdat = 856... was just a quick 200 line scan +-150mV... Will need a better one for sure
    dat855 was with 250mV SRS bias
    dat857 was with 150mV SRS bias
    dat858++ have 170mV SRS bias
    dats863+ was latest run. Up to 868
"""

def list_axes(fig: plt.Figure) -> List[plt.Axes]:
    for i, ax in enumerate(fig.axes):
        print(f'ax[{i}] -- {ax.title}')
    return fig.axes


def dcbias_width(dat, widths=[1,3,5,7]):
    dc = dat.DCbias
    fig, axs = PF.make_axes(1)
    plot_standard_dcbias(dat, axs, plots=[2])
    axs[0].lines[-1].remove()
    color = plt.cm.rainbow(np.linspace(0,1,len(widths)))
    for c, w in zip(color, widths):
        dat.DCbias.recalculate_fit(w)
        x1 = 14.1
        y1 = dc.full_fit.eval(x=x1)
        x2 = dc.get_current_for_target_heat(1.3)
        y2 = dc.full_fit.eval(x=x2)
        axs[0].scatter(x1, y1, s=500, marker='+', linewidths=1, c=c)
        axs[0].scatter(x2, y2, s=200, marker='x', linewidths=1, c=c)
        x = dc.full_fit.userkws['x']
        y = dc.full_fit.best_fit
        axs[0].plot(x, y, label=f'{w}nA', c=c)
        print(f'Fit_width={w}, dT at 5.4nA={dat.DCbias.get_dt_at_current(5.4):.3f}mV\n\t1.3*temp={dat.DCbias.get_current_for_target_heat(1.3):.2f}nA')
    axs[0].legend(title='Width of Fit')
    axs[0].grid('on')
    # PF.plot_dac_table(axs[1], dat)
    # plot_standard_dcbias(dat, [axs[1]], plots=[4])

def init_digamma(dat):
    if dat.Transition.fit_func != T.i_sense_digamma:
        dat.Transition.recalculate_fits(func=T.i_sense_digamma)
        print(f'dat{dat.datnum} Transition set to digamma')
        cfg.yes_to_all = True
        datdf.update_dat(dat)
        cfg.yes_to_all = False


def init_int_entropy(dat, recalc=False):
    if dat.Entropy.int_entropy_initialized is False or recalc is True:
        set_integrated_entropy(dat, dcdat)
        cfg.yes_to_all = True
        datdf.update_dat(dat)
        cfg.yes_to_all = False

if __name__ == '__main__':
    dcdat = make_dat_standard(772, dfoption='load') # 100mK = 772        # (862, 870) 50mK
    dats = [make_dat_standard(num, dfoption='sync') for num in [903, 905]] # range(879, 884+1)]    # 879-888     #[874, 878]  range(863, 868+1)] #[866, 867, 868]]
    # dcbias_width(dcdat, widths = [5,10, 15,20, 25])

    for dat in dats:
        if hasattr(dat, 'Entropy') is False:
            print(f'dat{dat.datnum} is not an entropy dat')
            continue
        if dat.Entropy.version != E.Entropy._Entropy__version:
            dat._reset_entropy()
        init_digamma(dat)
        init_int_entropy(dat, recalc=False)
        plot_integrated(dat)
    datdf.save()
    plot_nik_fit(dats)
    Dats.plot_transition_fits(dats)

    [print(f'Dat{dat.datnum}: RCT={dat.Logs.fdacs[4]:.0f}mV, dS_integrated = {dat.Entropy.int_ds:.3f}/kB, dS_fit = {dat.Entropy.dS:.3f}/kB, Amp = {dat.Transition.amp:.3f}nA') for dat in dats if hasattr(dat, 'Entropy')]

