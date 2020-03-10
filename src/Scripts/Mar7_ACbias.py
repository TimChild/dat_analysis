from src.Scripts.StandardImports import *
import src.Scripts.Mar3_Entropy_Vs_field as Mar3
import src.Scripts.Mar3_CompareDCbias as Mar3DC
from src.DatCode.Li_theta import plot_standard_li_theta

def plot_data_vs_bias(dats, data:str = 'Entropy', show_integrated=False, x_axis = 'nA'):
    fig, axs = PF.make_axes(1)

    ax = axs[0]
    ax_setup(ax, title=f'Dats{dats[0].datnum} to {dats[-1].datnum}: {data} vs Bias', x_label=f'Bias /{x_axis}')
    if x_axis.lower() == 'mv':
        x = [dat.Instruments.srs1.out for dat in dats]
    elif x_axis.lower() == 'na':
        x = [dat.Instruments.srs1.out/50*np.sqrt(2) for dat in dats]

    if data.lower() == 'entropy':
        y = [dat.Entropy.dS for dat in dats]
        y_err = [np.std(dat.Entropy.fit_values.dSs) for dat in dats]
        ax.set_ylabel('Entropy /kB')
    elif data.lower() == 'dt':
        y = [dat.Entropy.dT for dat in dats]
        y_err = None
        ax.set_ylabel('dT /mV')
    else:
        print(f'data must be in [entropy]')

    ax.errorbar(x, y, yerr=y_err, linestyle='None', marker='x', label = 'Fit')

    if show_integrated is True:
        y_int = [dat.Entropy.int_ds for dat in dats]
        y_int_err = [np.std(dat.Entropy.int_entropy_per_line[-1]) for dat in dats]
        ax.errorbar(x, y_int, yerr=y_int_err, marker='x', linestyle='None', label='Integrated')
        ax.legend()



def make_li_theta_int_entropy(dats):
    for dat in dats:
        dat.datname='li_dt'
        li_dats = make_dats(list(range(dat.datnum-2, dat.datnum)))
        dt = li_dats[1].Li_theta.fit_values.theta-li_dats[0].Li_theta.fit_values.theta
        dat.Entropy.init_integrated_entropy_average(dT_mV=dt, dT_err=0, amplitude=dat.Transition.avg_fit_values.amps[0],
                                                    amplitude_err=0)
        datdf.update_dat(dat, yes_to_all=True)
    datdf.save()


def plot_li_theta_entropy(dats):
    plot_data_vs_bias(dats, show_integrated=True, x_axis='nA')
    fig = plt.gcf()
    PF.add_standard_fig_info(fig)
    PF.add_to_fig_text(fig, 'Using dT from Li_theta measurements')


# dats = make_dats(list(range(1446, 1486+1)))  # All dats including srs theta measurements
# dats = [1448, 1451, 1454, 1457, 1460, 1463, 1466, 1471, 1474, 1477, 1480, 1483, 1486]  # Entropy dats
# dats = make_dats(dats)
# dc = make_dat_standard(1467, dfoption='load')

# dats = make_dats(list(range(1567, 1634+1)), dfoption='load')  # Zero mag field data
# dats = [1569, 1572, 1575, 1578, 1581, 1584, 1587, 1592, 1595, 1598, 1601, 1604, 1607, 1610, 1615, 1618, 1621, 1624, 1627, 1630, 1633]  # Entropy only from zero mag field
# dats = make_dats(dats)  # Entropy only dats
# dcs = make_dats([1588, 1611, 1634])

# dats = make_dats(list(range(1636, 1662+1)))  # -100, 50, 200mT AC bias
dats = [1636, 1637, 1638, 1639, 1640, 1641, 1642, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1654, 1655, 1656, 1657, 1658, 1659, 1660]  # Entropy only for -100, 50, 200mT
dats = make_dats(dats)
dcs = make_dats([1643, 1652, 1661])

for field, dc in zip([-100, 50, 200], dcs):
    field_dats = [dat for dat in dats if np.isclose(dat.Instruments.magy.field, field, atol=10)]
    for dat in field_dats:
        Mar3.init_int_entropy(dat, recalc=False, dcdat=dc, update=True)
    plot_data_vs_bias(field_dats, show_integrated=True)
    fig = plt.gcf()
    fig.suptitle(f'Field={field}mT')
    PF.add_standard_fig_info(fig)

# for dat in dats:
#     Mar3.init_int_entropy(dat, recalc=False, dcdat=dcs[2], update=True)

# plot_data_vs_bias(dats)


