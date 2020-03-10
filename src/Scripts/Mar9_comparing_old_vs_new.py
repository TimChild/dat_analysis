from src.Scripts.Mar7_Along_transition import plot_entropy_with_mid
from src.Scripts.StandardImports import *


def print_dat_info(dats):
    for dat in dats:
        print(f'Dat{dat.datnum}:')
        print(f'\tSRS1: {dat.Instruments.srs1}')
        print(f'\tSRS3: {dat.Instruments.srs3}')
        print(f'\tMax Entropy Y signal raw: {np.nanmax(dat.Data.ADC2_2d):.3f}')
        print(f'\tMax Entropy R filtered:{np.nanmax(dat.Entropy._data):.3f}nA')
        print(f'\tTemp: {dat.Logs.mc_temp:.3f}mK')
        print(f'\tSweeprate: {dat.Logs.sweeprate:.1f}mV/s')
        print(f'\tScan width: {dat.Data.x_array[-1]-dat.Data.x_array[0]:.1f}mV')
        print(f'\tTransition Fit values: {dat.Transition.avg_fit_values}')
        print(f'\tEntropy Fit values: {dat.Entropy.avg_fit_values}')
        # print(f'\t: {}')

    dfs = []
    for dat in dats:
        dn = dat.Logs.dacnames
        dacs = {dn.get(k, k): v for k, v in dat.Logs.dacs.items()}
        dn = dat.Logs.fdacnames
        fdacs = {dn.get(k, f'f{k}'): v for k, v in dat.Logs.fdacs.items()}

        all_dacs = {**dacs, **fdacs}
        dfs.append(pd.DataFrame([all_dacs.values()], columns=list(all_dacs.keys()), index=[f'Dat{dat.datnum}']))

    full_df = pd.concat(dfs)
    print(full_df)


nds = [1663, 1664, 1665, 1666, 1667]
ndc = [1668, 1689]

od = make_dat_standard(1306, dfoption='load')
nd = make_dat_standard(1663, dfoption='load')

DU.fix_logs(od)
dats = [od, nd]

d = make_dat_standard(1684)
plot_entropy_with_mid(d)



