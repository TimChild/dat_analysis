from src.Scripts.StandardImports import *
from src.DatCode.Entropy import plot_standard_entropy
import lmfit as lm
import copy

"""For plotting things with magnet field sweeps (Before fixing broken magnet jsons)"""

from src.Configs.Jan20Config import add_mag_to_logs


def init_int_entropy(dat, recalc=False, dcdat=None, update=True, savedf = False):
    # dcbias_dict = {
    #     200: {-550: 1327, -590: 1329},
    #     100: {-550: 1331, -590: 1333},
    #     50: {-550: 1335, -590: 1337},
    #     -50: {-550: 1339, -590: 1341}
    # }
    dcbias_dict = {
        200: {-550: 1373, -590: 1375},
        100: {-550: 1377, -590: 1379},
        50: {-550: 1381, -590: 1383},
        -50: {-550: 1385, -590: 1387},
        -100: {-550: 1389, -590: 1391},
        -200: {-550: 1393, -590: 1395}
    }
    if dat.Entropy.int_entropy_initialized is False or recalc is True:
        field = round(dat.Instruments.magy.field)
        hqpc = dat.Logs.fdacs[0]

        if dcdat is None:
            if field not in dcbias_dict.keys() or hqpc not in [-550, -590]:
                print(f'Dat{dat.datnum} has field = {field}mT, hqpc = {hqpc}mV which is not in dcbias_dict')
                return
            dcdat = make_dat_standard(dcbias_dict[field][hqpc], dfoption='load')
        elif type(dcdat) == int:
            dcdat = make_dat_standard(dcdat, dfoption='load')
        else:
            pass  # Just use passed in dcdat
        dt = dcdat.DCbias.get_dt_at_current(dat.Instruments.srs1.out / 50 * np.sqrt(2))
        dat.Entropy.init_integrated_entropy_average(dT_mV=dt/2, dT_err=0, amplitude=dat.Transition.avg_fit_values.amps[0],
                                                    amplitude_err=0, dcdat=dcdat)
        if update is True:
            datdf.update_dat(dat, yes_to_all=True)
            if savedf is True:
                datdf.save()
                print(f'Dat{dat.datnum} saved in df')
            else:
                print(f'dat{dat.datnum}[{dat.datname}] updated - need to save df')


def plot_average_entropys(dats):
    # add_mag_to_logs(dats)
    fig, axs = PF.make_axes(len(dats) + 1)
    for dat, ax in zip(dats, axs):
        if 'entropy' not in dat.dattype:
            print(f'dat{dat.datnum} has no entropy')
            continue
        plot_standard_entropy(dat, [ax], plots=[2], kwargs_list=[{'no_datnum': False}])
        ax.collections[0].set_sizes([1] * len(dat.Data.x_array))
        ax.set_title(f'Field = {dat.Instruments.magy.field:.1f}mT')
        ax.set_ylim(-0.004, 0.009)
        PF.ax_text(ax, f'dS={dat.Entropy.avg_fit_values.dSs[0]:.3f}')
    PF.plot_dac_table(axs[-1], dats[0])
    PF.add_standard_fig_info(fig)
    fig.suptitle('Average Entropy R vs Field')



def plot_full_entropy_info(dat):
    fig, axs = PF.make_axes(6)
    if dat.Entropy.int_entropy_initialized is False:
        print(f'Need to initialize int entropy on dat{dat.datnum}')
        return None
    plot_standard_entropy(dat, axs, plots=[1, 2, 3, 4, 10, 11])
    fig.suptitle(f'Dat{dat.datnum}: Field={dat.Instruments.magy.field:.0f}mT, HQPC={dat.Logs.fdacs[0]}mV')


def plot_multiple_entropy_fits(dat, y_indexes: list, ax=None, align=False, spacing_x=0, spacing_y=0):
    """Plots up to 10 1D datasets and fits on single axes with legend"""
    if ax is None:
        fig, ax = PF.make_axes(1)
        ax = ax[0]
    ax.set_title(f'Dat{dat.datnum}: Entropy Data and Fits')

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    i = 0
    for yid, c in zip(y_indexes, colors):
        x = dat.Entropy.x_array
        data = dat.Entropy._data[yid]
        fit = dat.Entropy._full_fits[yid]
        best_fit = fit.best_fit
        if align is True:
            x = x - fit.best_values['mid'] + spacing_x * i
            data = data - fit.best_values['const'] + spacing_y * i
            best_fit = best_fit - fit.best_values['const'] + spacing_y * i
            i += 1
        ax.scatter(x, data, marker=',', s=1, color=c)
        ax.plot(x, best_fit, linewidth=1, color=c, label=f'{yid}: {dat.Entropy.fit_values.dSs[yid]:.3f}')
    ax.legend(title='y_index: dS_fit')


def CS_bias_dats():
    dats = [make_dat_standard(num, dfoption='load') for num in [1399, 1400, 1401]]
    for dat in dats:
        if dat.Transition.version != '2.2':
            print(f'Recalculating fit for dat{dat.datnum}')
            dat.Transition.recalculate_fits()
        plot_full_entropy_info(dat)
        fig = plt.gcf()
        fig.suptitle(
            f'Dat{dat.datnum}: CSbias={-(dat.Logs.dacs[15] + 600) / 13.15:.0f}uV, Field={dat.Instruments.magy.field:.0f}mT, HQPC={dat.Logs.fdacs[0]}mV')


def refit_entropy_new_params(dat, params, new_name=None, update_df=True):
    """
    Takes single lm.Parameters as initial and refits by row and average Entropy fits

    @param dat:
    @type dat:  src.DatCode.Dat.Dat
    @param params: single lm.Parameters
    @type params: lm.Parameters
    @param new_name: new name of dat to save in datdf
    @type new_name: str
    @param update_df: whether to add to df automatically or not, defaults to True
    @type update_df: bool
    @return: new dat
    @rtype: src.DatCode.Dat.Dat
    """

    if new_name is not None:
        dat_old = dat
        dat = copy.deepcopy(dat_old)
        dat.datname = new_name
    dat.Entropy.recalculate_fits([params] * len(dat.Data.y_array))
    if update_df is True:
        datdf.update_dat(dat, yes_to_all=True)
    return dat


datnums = list(range(1290, 1300 + 1))
datnums4 = list(range(1306, 1327, 2))  # All 200mT, 40nA max HQPC -550mV
# Didn't recored entropy data for these... datnums5 = list(range(1328, 1340, 2))  # Changing HQPC between -550 and -590, and Fields through [200, 100, 50, -50]mT
datnums6 = list(range(1342, 1347 + 1))  # Changing HQPC between -550 and -590, and Fields through [200, 100, 50, -50]mT
datnums7 = list(range(1374, 1396 + 1, 2))  # same as above with more repeats and -100, -200mT as well.
datnums8 = list(range(1490, 1566+1))  # All dats in scan along transition 6th March  1529 is the only DCbias dat here
d8_entropy_dats = [1492, 1495, 1498, 1501, 1504, 1507, 1510, 1513, 1516, 1519, 1522, 1525, 1528, 1533, 1536, 1539, 1542, 1545,
       1548, 1551, 1554, 1557, 1560, 1563, 1566]  # Just the entropy scans of datnums8

# dats = [make_dat_standard(num, dfoption='load') for num in datnums7]


from src.DatCode.Transition import i_sense_digamma
def make_digamma(dats):
    for dat in dats:
        dat.datname = 'digamma'
        hdfpath = dat.Data.hdfpath
        dat.Logs.set_hdf_path(hdfpath)
        dat.Transition.recalculate_fits(func=i_sense_digamma)
        datdf.update_dat(dat, yes_to_all=True)
    datdf.save()

if __name__ == '__main__':
    dats = [make_dat_standard(num, datname='digamma', dfoption='load') for num in d8_entropy_dats]
    dc = make_dat_standard(1529, dfoption='load')



    # CS_bias_dats()
    # plot_average_entropys(dats)
    # for dat in dats:
    #     init_int_entropy(dat, recalc=False)
    #
    # for dat in dats[2:6]:
    #     plot_full_entropy_info(dat)
    pass