from src.Scripts.StandardImports import *
import src.Scripts.Mar3_Entropy_Vs_field as Mar3


def fix_logs(dat, update=True):
    if not hasattr(dat.Logs, 'version'):
        dat.Logs.set_hdf_path(dat.Data.hdfpath)
        print(f'Dat{dat.datnum}: added hdfpath to Logs and set version - DFupdated but not saved')
        if update is True:
            datdf.update_dat(dat, yes_to_all=True)


def fix_entropy_scale(dat, update=True):
    """For a while I was reading entropy with SRS1, but correcting based on the sensitivity of srs3 instead.."""
    fix_logs(dat, update=True)  # Might as well fix this while I'm here...
    if 'entropy' in dat.dattype:
        ratio = np.nanmax(dat.Data.ADC2_2d[0]) / np.nanmax(dat.Data.enty[0])
        if dat.Instruments.srs1.sens != 50:
            print(f'WARNING - NO ACTION TAKEN: Dat{dat.datnum}: SRS1sens = {dat.Instruments.srs1.sens}')
            return None
        if dat.Instruments.srs3.sens != 200:
            print(f'WARNING - NO ACTION TAKEN: Dat{dat.datnum}: SRS3sens = {dat.Instruments.srs3.sens}')
            return None
        if not np.isclose(ratio, 200, atol=10):
            multiplier = dat.Instruments.srs1.sens / 10 * 1e-3 / 1e9 * 1e9
            dat.Data.entx = dat.Data.entropy_x_2d[:] * multiplier
            dat.Data.enty = dat.Data.entropy_y_2d[:] * multiplier
            dat._reset_entropy()
            if update is True:
                datdf.update_dat(dat, yes_to_all=True)
            print(f'Dat{dat.datnum} - Old ratio ADC2/enty = {ratio:.1f}, now corrected')
        else:
            print(f'Dat{dat.datnum} - old ratio was {ratio:.1f} which sounds OK? Did not update anything')
        return None
    else:
        print(f'Dat{dat.datnum} is not an Entropy dat, only fixed logs if it was necessary.')


def entropy_comparison():
    """Temporary comparison to figure out what was going on with integrated entropy scaling"""
    # old dat = 1380, old dcdat = 1379
    # new dat = 1498, new dcdat = 1529

    od = make_dat_standard(1380, dfoption='load')
    odc = make_dat_standard(1379, dfoption='load')
    nd = make_dat_standard(1498, dfoption='load')
    ndc = make_dat_standard(1529, dfoption='load')

    dats = [od, nd]
    dcs = [odc, ndc]

    for dat in dats + dcs:
        fix_logs(dat)

    fig, axs = PF.make_axes(6)

    ax = axs[0]
    ax.set_title('Raw Entropy Y[0]')
    for dat in dats:
        ax.plot(dat.Data.x_array, dat.Data.ADC2_2d[0], label=f'dat{dat.datnum}')
    ax.set_xlabel(dat.Logs.x_label)
    ax.set_ylabel('Raw Entropy Signal')
    ax.legend()

    ax = axs[1]
    ax.set_title('Entropy Y[0] in nA')
    for dat in dats:
        ax.plot(dat.Data.x_array, dat.Data.enty[0], label=f'dat{dat.datnum}')
    ax.set_xlabel(dat.Logs.x_label)
    ax.set_ylabel('Entropy Signal /nA')
    ax.legend()

    ax = axs[2]
    ax.set_title('Integrated Entropy Average')
    for dat, dc in zip(dats, dcs):
        Mar3.init_int_entropy(dat, recalc=True, dcdat=dc, update=True)
        ax.plot(dat.Entropy.x_array, dat.Entropy.integrated_entropy, label=f'Dat{dat.datnum}')
    ax.set_xlabel(dat.Logs.x_label)
    ax.set_ylabel('Entropy /kB')
    ax.legend()

    ax = axs[3]
    ax.set_title('DCbias fit with Dat bias')
    for dc, dat in zip(dcs, dats):
        ax.plot(dc.DCbias.x_array_for_fit, dc.DCbias.full_fit.best_fit, label=f'Dat{dc.datnum}')
        ax.axvline(dat.Instruments.srs1.out / 50 * np.sqrt(2), color='k', linestyle=':',
                   label=f'Dat{dat.datnum} AC bias')
        ax.axvline(-dat.Instruments.srs1.out / 50 * np.sqrt(2), color='k', linestyle=':')
    ax.set_xlabel(dc.Logs.x_label)
    ax.set_ylabel('Theta /mV')
    ax.legend()

    ax = axs[4]
    ax.set_title('I_sense in nA')
    for dat in dats:
        ax.plot(dat.Data.x_array, dat.Data.i_sense[0], label=f'dat{dat.datnum}')
    ax.set_xlabel(dat.Logs.x_label)
    ax.set_ylabel('Current /nA')
    ax.legend()

    ax = axs[5]
    ax.set_title('Raw i_sense signal')
    for dat in dats:
        ax.plot(dat.Data.x_array, dat.Data.ADC0_2d[0], label=f'dat{dat.datnum}')
    ax.set_xlabel(dat.Logs.x_label)
    ax.set_ylabel('Current /?')
    ax.legend()

    plt.tight_layout()

    for dat in dats:
        print(f'Dat{dat.datnum}:\n'
              f'\tSRS1out:{dat.Instruments.srs1.out}\n'
              f'\tSRS1sens:{dat.Instruments.srs1.sens}\n'
              f'\tSRS3sens:{dat.Instruments.srs3.sens}\n'
              f'\tTransition Amp:{dat.Transition.avg_fit_values.amps[0]:.4f}nA')


d8_entropy_dats = [1492, 1495, 1498, 1501, 1504, 1507, 1510, 1513, 1516, 1519, 1522, 1525, 1528, 1533, 1536, 1539, 1542,
                   1545, 1548, 1551, 1554, 1557, 1560, 1563, 1566]


for datnum in range(1400, 1635):
    try:
        for name in datdf.df.loc[datnum].index.values:
            dat = make_dat_standard(datnum, datname=name, dfoption='load')
            fix_entropy_scale(dat)
    except Exception as e:
        print(f'Dat{datnum} threw exception [{e}]')
        continue
datdf.save()