"""This is for looking at how well measuring theta works with the lock-in"""
from src.Scripts.StandardImports import *
from src.DatCode.Li_theta import plot_standard_li_theta


def plot_multiple_li_theta(dats):
    fig, axs = PF.make_axes(len(dats))
    fig.suptitle(f'Lock-in Theta calculation from fits')

    for dat, ax in zip(dats, axs):
        plot_standard_li_theta(dat, [ax], plots=[1], kwargs_list=[{'show_eqn': False}])
        ax_setup(ax,
                 f'Dat{dat.datnum}: SRS1={dat.Instruments.srs1.out / 50 * np.sqrt(2):.2f}nA, By={dat.Instruments.magy.field:.0f}mT', fs = 8, legend=False)
    PF.add_standard_fig_info(fig)

def plot_li_theta_vs_fit_theta(li_dats, transition_dats=[], x_axis='Datnum'):
    fig, ax = PF.make_axes(1)
    ax = ax[0]
    if x_axis.lower() == 'datnum':
        x_li = [dat.datnum for dat in li_dats]
        x_trans = [dat.datnum for dat in transition_dats]
    else:
        print('x_axis must be in [datnum]')
        return None

    y_li = [dat.Li_theta.fit_values.theta for dat in li_dats]
    y_li_err = [dat.Li_theta.params['theta'].stderr for dat in li_dats]

    y_trans = [dat.Transition.avg_fit_values.thetas[0] for dat in transition_dats]
    y_trans_err = [np.std(dat.Transition.fit_values.thetas) for dat in transition_dats]

    ax.errorbar(x_li, y_li, yerr=y_li_err, linestyle='None', marker='x', label='Li_thetas')
    ax.errorbar(x_trans, y_trans, yerr=y_trans_err, linestyle='None', marker='x', label='Transition_thetas')

    ax_setup(ax, title=f'Theta vs {x_axis}', x_label=x_axis, y_label='Theta /mV', legend=True)
    PF.add_standard_fig_info(fig)
    return


def chunks(lst, n):
    """Break list into chunks of size n"""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


# dats = make_dats(list(range(1400, 1635)))
# ds = [dat for dat in dats if 'li_theta' in dat.dattype]
dats = [1431, 1432, 1433, 1434, 1442, 1443, 1446, 1447, 1449, 1450, 1452, 1453, 1455, 1456, 1458, 1459, 1461, 1462, 1464, 1465, 1469, 1470, 1472, 1473, 1475, 1476, 1478, 1479, 1481, 1482, 1484, 1485, 1487, 1488, 1490, 1491, 1493, 1494, 1496, 1497, 1499, 1500, 1502, 1503, 1505, 1506, 1508, 1509, 1511, 1512, 1514, 1515, 1517, 1518, 1520, 1521, 1523, 1524, 1526, 1527, 1531, 1532, 1534, 1535, 1537, 1538, 1540, 1541, 1543, 1544, 1546, 1547, 1549, 1550, 1552, 1553, 1555, 1556, 1558, 1559, 1561, 1562, 1564, 1565, 1567, 1568, 1570, 1571, 1573, 1574, 1576, 1577, 1579, 1580, 1582, 1583, 1585, 1586, 1590, 1591, 1593, 1594, 1596, 1597, 1599, 1600, 1602, 1603, 1605, 1606, 1608, 1609, 1613, 1614, 1616, 1617, 1619, 1620, 1622, 1623, 1625, 1626, 1628, 1629, 1631, 1632]
dats = make_dats(dats)

df = datdf.df[:]
df = df[df.dat_types.transition == True].loc[1425:]  # All transition dats from 1400 onwards
df = df[df.dat_types.dcbias != True]
trans_nums = np.array(df.index.to_list())[:,0].astype(int)

trans_dats = make_dats(trans_nums)



plot_li_theta_vs_fit_theta(dats, transition_dats=trans_dats)

for ds in chunks(dats, 9):
    plot_multiple_li_theta(ds)
