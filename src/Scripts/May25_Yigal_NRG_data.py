import scipy.io as sio
import os
from typing import List
from src.Scripts.StandardImports import *
import matplotlib.colors

path = os.path.normpath(r'D:\OneDrive\UBC LAB\My work\My_Papers\2Resources\Data\Yigal')
full_path1 = os.path.join(path, 'NRG gamma = 0.001.mat')
full_path2 = os.path.join(path, 'NRG gamma = 0.002.mat')

fp = full_path2
gamma = 0.002

data = sio.loadmat(fp)
y = data['Ens'][:, 0]
y_label = 'Energies'

t = data['Ts'][:, 0]
t_label = 'Temperature'

cond = data['Conductance_mat']
cond_label = 'Conductance'

entropy = data['Entropy_mat']
entropy_label = 'Entropy'

occ = data['Occupation_mat']
occ_label = 'Occupation'

dndt = data['DNDT_mat']
dndt_label = 'dN/dT'

int_dndt = data['intDNDT_mat']
int_dndt_label = 'Integrated dN/dT'

# d_occ = np.gradient(np.gradient(occ)[1])[0]
# d_occ_label = 'Diff Occ'

datas = [cond, entropy, occ, dndt, int_dndt]
titles = [cond_label, entropy_label, occ_label, dndt_label, int_dndt_label]


# lims = [(0, 0.175), (0, 1), (0, 1), (-10, 20), (0, 1), None]


def plot_2d(datas, t, gamma, axs=None, titles=None, lims=None, x_type='g/t'):
    if x_type == 't':
        x = t
        x_label = 'Temperature'
    elif x_type == 'g/t':
        x = np.array(gamma / t)
        x_label = f'{Char.GAMMA}/T'
    else:
        raise ValueError('x_type must be in ["t", "g/t"]')

    if lims is None:
        lims = [None] * len(datas)
    elif len(lims) != len(datas):
        raise ValueError(f'len(lims) = {len(lims)} != len(datas) = {len(datas)}')

    if axs is None:
        fig, axs = PF.make_axes(len(datas), single_fig_size=(4, 4), plt_kwargs={'sharex': False, 'sharey': False})

    if titles is None:
        titles = [''] * len(datas)

    for data, title, ax, lim in zip(datas, titles, axs, lims):
        if lim is not None:
            norm = mpl.colors.Normalize(vmin=lim[0], vmax=lim[1])
        else:
            norm = None
        xx, yy = np.meshgrid(x, y)
        ax.pcolormesh(xx, yy, data, norm=norm)
        # PF.display_2d(x, y, data, ax, colorscale=True, x_label=x_label, y_label=y_label, auto_bin=False, norm=norm)
        ax.set_xlim(min(x), max(x))
        ax.set_xscale('log')
        PF.ax_setup(ax, title, x_label, y_label)
        plt.tight_layout()


def plot_1d(xloc, axs: List[plt.Axes], datas, t, titles=None, x_id_type='t', replace=True):
    if x_id_type == 't':
        xval = xloc
        leg_str = 'Temperature'
    elif x_id_type == 'g/t':
        xval = gamma / xloc
        leg_str = f'{Char.GAMMA}/T'
    else:
        raise ValueError('x_id_type must be ["t", "g/t"] for temp or gamma/t ratio')

    if titles is None:
        titles = ['']*len(datas)

    id = CU.get_data_index(t, xval)
    for ax, data, title in zip(axs, datas, titles):
        if replace is True:
            ax.cla()
        PF.ax_setup(ax, title, 'Energy', )
        ax.plot(y, data[:, id], label=f'{id}, {CU.sig_fig(xloc, 2)}')
        ax.legend(title=f'Row, {leg_str}')
    fig = plt.gcf()
    fig.tight_layout()


def plot_slices(datas, t, axs1d, axs2d, loc=None, titles=None, type='t', replace=True):
    if len(axs2d[0].collections) == 0:  # if not already plotted then plot
        plot_2d(datas, t, gamma, axs=axs2d, titles=titles, lims=None, x_type=type)

    if (type == 't' and axs2d[0].get_xlabel() != 'Temperature') or (type == 'g/t' and axs2d[0].get_xlabel() != f'{Char.GAMMA}/T'):  # wrong x axis so re plot
        plot_2d(datas, t, gamma, axs=axs2d, titles=titles, lims=None, x_type=type)


    for ax in axs2d:
        if replace is True:
            try:
                ax.lines[-1].remove()
            except:
                pass
        if loc is not None:
            ax.axvline(loc, color='red', linestyle=':')

    if loc is not None:
        plot_1d(loc, axs1d, datas, t, titles, x_id_type=type, replace=replace)
