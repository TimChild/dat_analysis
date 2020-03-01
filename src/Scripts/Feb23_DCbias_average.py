from src.Scripts.StandardImports import *
import src.DatCode.DCbias as DC



def get_useful_dcbias():
    df = datdf.df.reset_index()

    d = df[
        (df[('dat_types', 'dcbias')] == True) &
        (df[('datnum', '')].between(200, 800))
        ]

    datnums = d['datnum']
    dats = [make_dat_standard(num, dfoption='load') for num in datnums]
    useful_dats = []
    for dat in dats:
        if np.isclose(dat.Logs.temp, 100, atol=2) and np.isclose(dat.Logs.fdacs[6], -230,
                                                                 atol=1) and dat.Logs.y_label != 'Repeats':
            if np.isclose(dat.Logs.fdacs[0], -515, atol=1):
                useful_dats.append(dat)

    return useful_dats


def replot(dat):
    axs = plt.gcf().axes
    DC.plot_standard_dcbias(dat, axs, plots=(1,2,3,4))


if __name__ == '__main__':

    dats = [make_dat_standard(num, dfoption='sync') for num in [277, 278, 313, 771, 772, 773, 774, 775]]  # [277, 278, 313, 771]
    for dat in dats[-4:]:
        if dat.DCbias.version != DC.DCbias._DCbias__version:
            dat._reset_dcbias()
            dat.DCbias.recalculate_fit(width=15)
            datdf.update_dat(dat)
            print(f'Updated dat{dat.datnum}. Need to save df')
        pf = dat.DCbias.standard_plot_function()
        fig, axs = PF.make_axes(4)
        pf(dat, axs, plots=(1,2,3,4))

    for dat in dats:
        print(f'Dat{dat.datnum}: dT at 14.1nA = {dat.DCbias.get_dt_at_current(14.1):.4f}mV')