import matplotlib.pyplot as plt
import numpy as np
import OLD.MarchMeeting.Nik_datatools as dt
from matplotlib import gridspec

fig = plt.figure(figsize=dt.mm2inch((89 * 2, 200)))

outer = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.4)

gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                       width_ratios=[1, 1], wspace=0.02)
ax1 = fig.add_subplot(gs1[0, 0])  # g_sens du=0
ax2 = fig.add_subplot(gs1[0, 1])  # g_sens du!=0


gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                       width_ratios=[1, 1], wspace=0.02)

ax3 = fig.add_subplot(gs2[0,0])
ax4 = fig.add_subplot(gs2[0,1])


gs3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2],
                                       width_ratios=[1, 1], wspace=0.02)

ax5 = fig.add_subplot(gs3[0, 0])  # g_sens du=0
ax6 = fig.add_subplot(gs3[0, 1])  # g_sens du!=0

V0 = 0.0  # mV
dV0 = 0.0  # mV
theta1 = 0.275  # mV
theta2 = 0.600  # mV
lin = 10e-4
xlim = (-6, 6)


xnew = np.linspace(*xlim, 500)
g1 = dt.i_sense(xnew, V0, theta1, 0.045, lin, 0.5)
g2 = dt.i_sense(xnew, V0 + dV0, theta2, 0.045, lin, 0.5)
ax1.plot(xnew, g1, c='C0')
ax1.plot(xnew, g2, c='C3')
ax1.fill_between(xnew, g1, g2, facecolor='C7', alpha=0.2)
ax1.vlines(V0, 0, 0.5, color='k', linestyle=":")
ax1.hlines(0.5, -50, V0, color='k', linestyle=":")

dV0 = -1.6 * 150 / 1000  # -1.6uV/mK*100mK in mV
g1 = dt.i_sense(xnew, V0, theta1, 0.045, lin, 0.5)
g2s = dt.i_sense(xnew, V0 + dV0, theta2, 0.045, lin, 0.5)
ax2.plot(xnew, g1, c='C0')
ax2.plot(xnew, g2s, c='C3')
ax2.fill_between(xnew, g1, g2s, facecolor='C7', alpha=0.2)
ax2.vlines(V0, 0, 0.5, color='k', linestyle=":")
ax2.vlines(V0 + dV0, 0, 0.5, color='k', linestyle=":")
ax2.hlines(0.5, -50, V0, color='k', linestyle=":")

ax1.set_xticks([V0])
ax1.set_xticklabels([r'V$_{mid}$'], rotation=50)

ax2.set_xticks([V0 + dV0, V0])
ax2.set_xticklabels([r'V$_{mid}$', r'V$_{mid}$'], rotation=50)
colors = ['C3', 'C0']
aligns = ['right', 'center']
for xtick, color, align in zip(ax2.get_xticklabels(), colors, aligns):
    xtick.set_color(color)
    xtick.set_horizontalalignment(align)

for a in [ax1, ax2]:
    a.set_ylim(0.45, 0.55)
    a.set_xlim(*xlim)
    a.set_yticklabels([])
    a.tick_params(axis='x', top=False)
    a.tick_params(axis='x', direction='out')
    a.tick_params(axis='y', right=False)
    a.set_xlabel(r'V$_{p}$', x=0.9, labelpad=-20)
    a.text(0.20, 0.30, r'$N-1$', transform=a.transAxes, fontsize=12, fontweight='bold')
    a.text(0.58, 0.30, r'$N$', transform=a.transAxes, fontsize=12, fontweight='bold')

ax1.set_ylabel(r'I$_{sens}$')








ax3.plot(xnew, -1.0 * (g2 - g1), c='C7')
ax3.fill_between(xnew, 0.0, -1.0 * (g2 - g1), facecolor='C7', alpha=0.2)
ax3.vlines(V0, -0.05, 0.0, color='k', linestyle=":")
ax3.hlines(0.0, -50, 50, color='k', linestyle=":")
ax3.set_xticks([V0])
# inset_ax1.set_xticklabels([r'V$_{mid}$'], rotation=50, fontsize=10)
ax3.set_ylabel(r'$\delta I_{sens}$', y=0.6, labelpad=0, fontsize=12)

ax4.plot(xnew, -1.0 * (g2s - g1), c='C7')
ax4.fill_between(xnew, 0.0, -1.0 * (g2s - g1), facecolor='C7', alpha=0.2)
ax4.vlines(V0, -0.05, 0.0, color='k', linestyle=":")
ax4.vlines(V0 + dV0, -0.050, 0.0, color='k', linestyle=":")
ax4.hlines(0.0, -50, 50, color='k', linestyle=":")
ax4.set_xticks([V0 + dV0, V0])

# inset_ax2.set_xticklabels([r'V$_{mid}$', r'V$_{mid}$'], rotation=50, fontsize=10)
# colors = ['C3', 'C0']
# aligns = ['right', 'center']
# for xtick, color, align in zip(inset_ax2.get_xticklabels(), colors, aligns):
#     xtick.set_color(color)
#     xtick.set_horizontalalignment(align)

for a in [ax3, ax4]:
    a.set_ylim(-0.024, 0.024)
    a.set_xlim(*xlim)
    a.set_yticklabels([])
    a.set_xticklabels([])
    a.tick_params(axis='x', top=False)
    a.tick_params(axis='x', direction='out')
    a.tick_params(axis='y', right=False, left=False)
    a.set_xlabel(r'V$_{p}$', x=0.9, labelpad=-5, fontsize=12)



dg1 = g1-g2
dg2 = g1-g2s
intg1 = np.cumsum(dg1)
intg2 = np.cumsum(dg2)
ax5.plot(xnew, intg1, c='C7')
ax6.plot(xnew, intg2, c='C7')
ax5.fill_between(xnew, 0.0, intg1, facecolor='C7', alpha=0.2)
ax6.fill_between(xnew, 0.0, intg2, facecolor='C7', alpha=0.2)
ax5.hlines(0.0, -50, 50, color='k', linestyle=":")
ax6.hlines(0.0, -50, 50, color='k', linestyle=":")
ax5.vlines(V0, 0, np.max(intg1), color='k', linestyle=":")
ax6.vlines(V0, 0, np.max(intg2), color='k', linestyle=":")
# ax6.vlines(V0+dV0, 0, np.max(intg2), color='k', linestyle=":")
ax5.set_xticks([V0])
ax6.set_xticks([V0 + dV0, V0])
ax5.set_ylabel(r'$\int \delta I_{sens}dV_{P}$', y=0.6, labelpad=0, fontsize=12)

for a in [ax5, ax6]:
    # a.set_ylim(-0.024, 0.024)
    a.set_xlim(*xlim)
    a.set_yticklabels([])
    a.set_xticklabels([])
    a.tick_params(axis='x', top=False)
    a.tick_params(axis='x', direction='out')
    a.tick_params(axis='y', right=False, left=False)
    a.set_xlabel(r'V$_{p}$', x=0.9, labelpad=-5, fontsize=12)







# with plt.rc_context({"text.usetex": True, "font.family": "serif", "font.serif": "cm"}):
# ax1.text(0.025, 0.05, r'$\partial S / \partial N = 0$',
#          transform=ax1.transAxes)
# ax2.text(0.025, 0.05, r'$\partial S / \partial N > 0$',
#          transform=ax2.transAxes)

# dt.add_subplot_id(ax0a, 'a', loc=(-0.15, 0.9))
# dt.add_subplot_id(ax1, 'b', loc=(-0.1, 1.05))
# dt.add_subplot_id(ax2, 'c', loc=(1.05, 1.05))
#
# fig.savefig(os.path.join(fig_dir, 'figure_1_no-annotation.pdf'), dpi=600)
