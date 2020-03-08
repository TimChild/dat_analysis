from src.Scripts.StandardImports import *
import src.DatCode.Entropy as E
import src.DatCode.DCbias as DC
"""This is for dats(1158-1218)(1060 - 1147 didn't work because an i was a j in one of the scan loops) 
 Looking entropy on right side with various HQPC settings and various ACbias for each
HQPC

HQPC values from -550 to -490mV (basically over the whole range of very closed off to just about at the first plateau)
And at each HQPC setting doing entropy scans with 100 to 700mV AC bias.


Also dats1361-1371 are ACbias scans with mag field
also dats1416 - 1423 are ACbias scans with mag field
"""


def plot_2d_entropy(dat, ax):
    fig = plt.gcf()
    if len(fig.texts) == 0:
        fig.suptitle('2D entropy R')
    PF.display_2d(dat.Entropy.x_array, dat.Data.y_array, dat.Entropy.entr, ax, dat=dat, x_label=None, y_label=None)


def plot_2d_transition(dat, ax):
    fig = plt.gcf()
    if len(fig.texts) == 0:
        fig.suptitle('2D transition')
    PF.display_2d(dat.Transition._x_array, dat.Data.y_array, dat.Transition._data, ax, dat=dat, x_label=None, y_label=None)


def plot_nik_entropy(dat, ax):
    fig = plt.gcf()
    if len(fig.texts) == 0:
        fig.suptitle('Nik Entropy')
    E.plot_standard_entropy(dat, [ax], plots=[4], kwargs_list=[{'no_datnum':False}])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')


def plot_avg_entropy(dat, ax):
    fig = plt.gcf()
    if len(fig.texts) == 0:
        fig.suptitle('Average Entropy R')
    E.plot_standard_entropy(dat, [ax], plots=[2], kwargs_list=[{'no_datnum':False}])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')


def plot_avg_integrated_entropy(dat, ax):
    fig = plt.gcf()
    if len(fig.texts) == 0:
        fig.suptitle('Average Integrated Entropy')
    if dat.Entropy.int_entropy_initialized is False:
        hqpc = dat.Logs.fdacs[0]
        dcdat = make_dat_standard(dcdats[hqpcs.index(hqpc)], dfoption='load')
        dt = dcdat.DCbias.get_dt_at_current(dat.Instruments.srs1.out/50*np.sqrt(2))
        dat.Entropy.init_integrated_entropy_average(dT_mV=dt, dT_err=0, amplitude=dat.Transition.amp, amplitude_err=0)
        cfg.yes_to_all = True
        datdf.update_dat(dat)
        cfg.yes_to_all = False
        print(f'dat{dat.datnum} Integrated entropy updated with dT={dt:.3f}mV, amp = {dat.Transition.amp:.3f}nA')
    E.plot_standard_entropy(dat, [ax], plots=[9], kwargs_list=[{'no_datnum':False}])
    ax.texts[-1].remove()
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')



def plot_array(hqpcs, acbias, plotfn, dats):
    fig, axs = plt.subplots(len(acbias), len(hqpcs), squeeze=False, sharex=True, sharey=True, figsize=(10, 10))

    for dat in dats:

        if 'entropy' in dat.dattype:
            try:
                row = acbias.index(dat.Instruments.srs1.out)
                column = hqpcs.index(dat.Logs.fdacs[0])
            except ValueError:
                continue
            ax = axs[row][column]
            ax: plt.Axes
            plotfn(dat, ax)

    for i, ax in enumerate(axs[:, -1]):
        ax.text(1.02, 0.5, f'{acbias[i]}mV', rotation=90, transform=ax.transAxes, verticalalignment='center', horizontalalignment='left', clip_on=False)
    for i, ax in enumerate(axs[0]):
        ax.set_title(f'HQPC={hqpcs[i]}')
    PF.add_standard_fig_info(fig)


def plot_all_dcbias(dcdats):
    dats = [make_dat_standard(num, dfoption='load') for num in dcdats]
    for dat in dats:
        plot_dcbias(dat)


def plot_dcbias(dat):
    fig, axs = PF.make_axes(4)
    DC.plot_standard_dcbias(dat, axs, plots=[1, 2, 3, 4], kwargs_list=[{}, {'marker': '.'}, {}, {}])
    PF.add_to_fig_text(fig, f'HQPC={dat.Logs.fdacs[0]}mV')

plotfn = plot_avg_integrated_entropy
# datnums = range(1158, 1215 + 1)
datnums = range(1361, 1371+1)
# datnums = [1060, 1061, 1062]
dats = [make_dat_standard(num, dfoption='load') for num in datnums]


hqpcs = [-550, -540, -530, -520, -510, -500, -490]
dcdats = [1158, 1166, 1174, 1182, 1190, 1198, 1206]
acbias = [100, 200, 300, 400, 500, 600, 700]

if __name__ == '__main__':
    # plot_array(hqpcs, acbias, plotfn, dats)
    # plot_all_dcbias(dcdats)
    pass