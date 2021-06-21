import matplotlib.pyplot as plt
import OLD.MarchMeeting.Nik_datatools as dt
from matplotlib import gridspec
import os
import numpy as np


### setup figure
fig = plt.figure(figsize=dt.mm2inch((89 * 2, 140)))

outer = gridspec.GridSpec(2, 1, height_ratios=[1.25, 1], hspace=0.2)

gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0],
                                       width_ratios=[1, 1, 1], wspace=0.05)
ax0a = fig.add_subplot(gs1[0, 0])  # image
ax0b = fig.add_subplot(gs1[0, 1])  # image
axE = fig.add_subplot(gs1[0, 2])  # label/annotation space
axE.axis('off')

gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                       width_ratios=[1, 1], wspace=0.02)
ax1 = fig.add_subplot(gs2[0, 0])  # g_sens du=0
ax2 = fig.add_subplot(gs2[0, 1])  # g_sens du!=0

############################
### display device image ###
############################
#
# # plot the same data on both axes
# ax0a.imshow(mpl.image.imread(os.path.join(fig_dir, 'device_cropped.png')))
# ax0b.imshow(mpl.image.imread(os.path.join(fig_dir, 'device_cropped.png')))
#
# ax0a.set_xlim(0, 200)
# ax0b.set_xlim(400, 600)
#
# # hide the spines between ax and ax2
# ax0a.spines['right'].set_visible(False)
# ax0b.spines['left'].set_visible(False)
# ax0a.yaxis.tick_left()
# ax0a.tick_params(labelright='off')
# ax0b.yaxis.tick_right()
#
# d = .015  # how big to make the diagonal lines in axes coordinates
# kwargs = dict(transform=ax0a.transAxes, color='k', clip_on=False)
# ax0a.plot((1 - d, 1 + d), (-d, +d), **kwargs)
# ax0b.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
# kwargs.update(transform=ax0b.transAxes)
# ax0b.plot((-d, +d), (1 - d, 1 + d), **kwargs)
# ax0b.plot((-d, +d), (-d, +d), **kwargs)
#
# for a in [ax0a, ax0b]:
#     a.set_xticks([])
#     a.xaxis.set_major_formatter(plt.NullFormatter())
#     a.set_yticks([])
#     a.yaxis.set_major_formatter(plt.NullFormatter())

###################
### fake g data ###
###################

V0 = -485.0  # mV
dV0 = 0.0  # mV
theta1 = 0.275  # mV
theta2 = 0.600  # mV

xnew = np.linspace(-500, -450, 500)
g1 = dt.i_sense(xnew, V0, theta1, 0.045, 10e-4, 0.5)
g2 = dt.i_sense(xnew, V0 + dV0, theta2, 0.045, 10e-4, 0.5)
ax1.plot(xnew, g1, c='C0')
ax1.plot(xnew, g2, c='C3')
ax1.fill_between(xnew, g1, g2, facecolor='C7', alpha=0.2)
ax1.vlines(V0, 0, 0.5, color='k', linestyle=":")
ax1.hlines(0.5, -500, V0, color='k', linestyle=":")

dV0 = -1.6 * 150 / 1000  # -1.6uV/mK*100mK in mV
g1 = dt.i_sense(xnew, V0, theta1, 0.045, 10e-4, 0.5)
g2s = dt.i_sense(xnew, V0 + dV0, theta2, 0.045, 10e-4, 0.5)
ax2.plot(xnew, g1, c='C0')
ax2.plot(xnew, g2s, c='C3')
ax2.fill_between(xnew, g1, g2s, facecolor='C7', alpha=0.2)
ax2.vlines(V0, 0, 0.5, color='k', linestyle=":")
ax2.vlines(V0 + dV0, 0, 0.5, color='k', linestyle=":")
ax2.hlines(0.5, -500, V0, color='k', linestyle=":")

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
    a.set_xlim(-488, -481)
    a.set_yticklabels([])
    a.tick_params(axis='x', top=False)
    a.tick_params(axis='x', direction='out')
    a.tick_params(axis='y', right=False)
    a.set_xlabel(r'V$_{p}$', x=0.9, labelpad=-20)
    a.text(0.20, 0.30, r'$N-1$', transform=a.transAxes, fontsize=12, fontweight='bold')
    a.text(0.58, 0.30, r'$N$', transform=a.transAxes, fontsize=12, fontweight='bold')

ax1.set_ylabel(r'G$_{sens}$')

####################
### fake dg data ###
####################

pcent = "45%"
bpad = 0.3

inset_ax1 = inset_axes(ax1, width=pcent, height=pcent, loc=1, borderpad=bpad)
inset_ax1.plot(xnew, -1.0 * (g2 - g1), c='C7')
inset_ax1.fill_between(xnew, 0.0, -1.0 * (g2 - g1), facecolor='C7', alpha=0.2)
inset_ax1.vlines(V0, -0.05, 0.0, color='k', linestyle=":")
inset_ax1.hlines(0.0, -500, -470, color='k', linestyle=":")
inset_ax1.set_xticks([V0])
# inset_ax1.set_xticklabels([r'V$_{mid}$'], rotation=50, fontsize=10)

inset_ax2 = inset_axes(ax2, width=pcent, height=pcent, loc=1, borderpad=bpad)
inset_ax2.plot(xnew, -1.0 * (g2s - g1), c='C7')
inset_ax2.fill_between(xnew, 0.0, -1.0 * (g2s - g1), facecolor='C7', alpha=0.2)
inset_ax2.vlines(V0, -0.05, 0.0, color='k', linestyle=":")
inset_ax2.vlines(V0 + dV0, -0.050, 0.0, color='k', linestyle=":")
inset_ax2.hlines(0.0, -500, -470, color='k', linestyle=":")
inset_ax2.set_xticks([V0 + dV0, V0])
# inset_ax2.set_xticklabels([r'V$_{mid}$', r'V$_{mid}$'], rotation=50, fontsize=10)
# colors = ['C3', 'C0']
# aligns = ['right', 'center']
# for xtick, color, align in zip(inset_ax2.get_xticklabels(), colors, aligns):
#     xtick.set_color(color)
#     xtick.set_horizontalalignment(align)

for a in [inset_ax1, inset_ax2]:
    a.set_ylim(-0.024, 0.024)
    a.set_xlim(-488, -482)
    a.set_yticklabels([])
    a.set_xticklabels([])
    a.tick_params(axis='x', top=False)
    a.tick_params(axis='x', direction='out')
    a.tick_params(axis='y', right=False)
    a.set_xlabel(r'V$_{p}$', x=0.9, labelpad=-5, fontsize=12)
    a.set_ylabel(r'$\delta G_{sens}$', y=0.6, labelpad=0, fontsize=12)

with plt.rc_context({"text.usetex": True, "font.family": "serif", "font.serif": "cm"}):
    ax1.text(0.025, 0.05, r'$\partial S / \partial N = 0$',
             transform=ax1.transAxes)
    ax2.text(0.025, 0.05, r'$\partial S / \partial N > 0$',
             transform=ax2.transAxes)

dt.add_subplot_id(ax0a, 'a', loc=(-0.15, 0.9))
dt.add_subplot_id(ax1, 'b', loc=(-0.1, 1.05))
dt.add_subplot_id(ax2, 'c', loc=(1.05, 1.05))

fig.savefig(os.path.join(fig_dir, 'figure_1_no-annotation.pdf'), dpi=600)