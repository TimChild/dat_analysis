from src.Scripts.StandardImports import *


def _quick_repeats_Mar12():
    """This was just some quick repeat measurements I took before starting the ramp to zero then measure....
    Showed entropy around 0.6 +- 0.05"""
    dats = make_dats(list(range(1868, 1877 + 1)))
    for dat in dats:
        params = dat.Entropy.avg_params
        if params['const'].vary is False:
            params['const'].vary = True
            dat.Entropy.recalculate_fits([params] * len(dat.Data.y_array))
            datdf.update_dat(dat, yes_to_all=True)
    datdf.save()


def _ramp_zero_repeats_one_gate(dats = make_dats(list(range(1878, 1931 + 1)))):
    """For dats 1878 to ???: This is ramping everything to zero and then back up to standard setting and immediately
    starting 5minute scan repeats to see how long it takes things to settle and to see whether entropy reliably returns
    to the same value"""


    # What number scan after ramping to zero, taking this from the i=# that I put in the comments which happens to be
    # in the [1] position
    positions = [int(dat.Logs.comments.split(',')[1][2:]) for dat in dats]

    # This is to make a list like 0,0,0,0,1,1,1,1,2,2,2,2, etc to see which repeat we are on for each dat
    repeat_num = []
    r = 0
    for dat, p in zip(dats, positions):
        repeat_num.append(r)
        if p == np.nanmax(positions):
            r += 1

    fig, axs = PF.make_axes(1)
    ax = axs[0]
    colors = PF.get_colors(np.nanmax(repeat_num)+1, 'tab10')
    for repeat, c in zip(range(np.nanmax(repeat_num)+1), colors):
        for dat, p, r in zip(dats, positions, repeat_num):
            if r == repeat:
                if p == 0:
                    t0 = pd.to_datetime(dat.Logs.time_completed)
                    x = pd.Timedelta(0).seconds / 60
                    ax.scatter(x, dat.Entropy.avg_fit_values.dSs[0], marker='x', color=c, label=f'Repeat[{repeat}]')
                else:
                    x = (pd.to_datetime(dat.Logs.time_completed) - t0).seconds / 60
                    ax.scatter(x, dat.Entropy.avg_fit_values.dSs[0], marker='x', color=c)
    ax_setup(ax, f'Entropy vs Time\nDats{dats[0].datnum} to {dats[-1].datnum}', 'Time /min', 'Entropy /kB', legend=True)
    PF.add_standard_fig_info(fig)


def _ramp_zero_repeats_multiple_gates(dats = make_dats(list(range(1932, 2042+1))),  fdac_stepped='RCT'):
    """For dats 1932 to ????: This is ramping everything to zero and then back up to standard setting and immediately
    starting 5minute scan repeats to see how long it takes things to settle and to see whether entropy reliably returns
    to the same value
    Then repeating above at a different gate value and repeating etc"""

    if fdac_stepped == 'RCT':
        fdac_channel = 4
    else:
        raise ValueError

    all_dats = dats
    gate_values = sorted({dat.Logs.fdacs[fdac_channel] for dat in all_dats})  # Set so only contains each value once (has to be exact)

    fig, axs = PF.make_axes(len(gate_values), plt_kwargs=dict(sharex=True, sharey=True))
    for gv, ax in zip(gate_values, axs):
        dats = [dat for dat in all_dats if dat.Logs.fdacs[fdac_channel] == gv]

        # What number scan after ramping to zero, taking this from the i=# that I put in the comments which happens to be
        # in the [1] position
        positions = [int(dat.Logs.comments.split(',')[1][2:]) for dat in dats]

        # This is to make a list like 0,0,0,0,1,1,1,1,2,2,2,2, etc to see which repeat we are on for each dat
        repeat_num = []
        r = 0
        for dat, p in zip(dats, positions):
            repeat_num.append(r)
            if p == np.nanmax(positions):
                r += 1

        colors = PF.get_colors(np.nanmax(repeat_num)+1, 'tab10')
        for repeat, c in zip(range(np.nanmax(repeat_num)+1), colors):
            for dat, p, r in zip(dats, positions, repeat_num):
                if r == repeat:
                    if p == 0:
                        t0 = pd.to_datetime(dat.Logs.time_completed)
                        x = pd.Timedelta(0).seconds/60
                        ax.scatter(x, dat.Entropy.avg_fit_values.dSs[0], marker='x', color=c, label=f'Repeat[{repeat}]\nDat{dat.datnum}+')
                    else:
                        x = (pd.to_datetime(dat.Logs.time_completed)-t0).seconds/60
                        ax.scatter(x, dat.Entropy.avg_fit_values.dSs[0], marker='x', color=c)
        ax_setup(ax, f'{fdac_stepped}={gv}', legend=True)
    fig.text(0.5, 0.1, 'Time /min', ha='center', fontsize=14)
    fig.text(0.02, 0.5, 'Entropy /kB', va='center', rotation='vertical', fontsize=14)
    fig.suptitle('Entropy Vs Time at Various RCT with no moving between scans')
    PF.add_standard_fig_info(fig)


def _entropy_vs_time(dats = make_dats(list(range(1932, 2042+1))), fdac_stepped=None):
    """Entropy vs time with color representing different gate voltages"""

    all_dats = dats
    if fdac_stepped is None:
        gate_values = [None]
    else:
        if fdac_stepped == 'RCT':
            fdac_channel = 4
        else:
            raise ValueError
        gate_values = sorted({dat.Logs.fdacs[fdac_channel] for dat in
                              all_dats})  # Set so only contains each value once (has to be exact)

    fig, axs = PF.make_axes(1)
    ax = axs[0]
    colors = PF.get_colors(len(gate_values), 'tab10')
    t0 = pd.to_datetime(dats[0].Logs.time_completed)
    for dat in dats:
        if gate_values[0] is not None:
            c = colors[gate_values.index(dat.Logs.fdacs[fdac_channel])]
        else:
            c = colors[0]
        x = (pd.to_datetime(dat.Logs.time_completed)-t0).seconds/60
        ax.scatter(x, dat.Entropy.avg_fit_values.dSs[0], marker='x', color=c)

    if gate_values[0] is not None:
        for gv, c in zip(gate_values, colors):
            plt.scatter([], [], c=c, s=10, marker='x', label=f'{gv}mV')
        plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title=f'{fdac_stepped}')

    ax_setup(ax, f'Entropy Vs Time: Dats{all_dats[0].datnum} to {all_dats[-1].datnum}', 'Time /min', 'Entropy /kB')
    PF.add_standard_fig_info(fig)


# _ramp_zero_repeats_multiple_gates()
# _entropy_vs_time(dats = make_dats(list(range(2047, 2160+1))), fdac_stepped='RCT')
# _ramp_zero_repeats_multiple_gates(dats=make_dats(list(range(2047, 2160+1))), fdac_stepped='RCT')

# _entropy_vs_time(dats = make_dats(list(range(2166, 2785+1))), fdac_stepped='RCT')
# _ramp_zero_repeats_multiple_gates(dats=make_dats(list(range(2166, 2785+1))), fdac_stepped='RCT')

_entropy_vs_time(dats=make_dats(list(range(2786, 2915+1))))