"""
Sep 21 -- This was used throughout most measurements to plot all sorts of things for scans going from weakly to strongly
coupled.

Too messy to be useful in future, better off just starting from scratch if I need to do any of this again.
"""
import dat_analysis.hdf_util
import dat_analysis.useful_functions as U
import dat_analysis.dat_analysis.characters as C
from dat_analysis.dat_object.make_dat import get_dat, get_dats
from dat_analysis.plotting.plotly.dat_plotting import OneD, TwoD
from Analysis.Feb2021.common import set_sf_from_transition
from dat_analysis.analysis_tools.transition import do_transition_only_calc
from dat_analysis.analysis_tools.entropy import do_entropy_calc, dat_integrated_sub_lin
from dat_analysis.analysis_tools.csq_mapping import setup_csq_dat, calculate_csq_map
from dat_analysis.plotting.plotly.common_plots.transition import transition_trace, single_transition_trace, transition_fig
from dat_analysis.plotting.plotly.common_plots.entropy import entropy_vs_time_trace, entropy_vs_time_fig, get_integrated_trace, \
    get_integrated_fig
from dat_analysis.analysis_tools.general_fitting import FitInfo

from progressbar import progressbar
import pandas as pd
import logging
import lmfit as lm
import plotly.io as pio
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from dat_analysis.dat_object.make_dat import DatHDF

pio.renderers.default = 'browser'
logger = logging.getLogger(__name__)


def dats_to_dT_df(dats: List[DatHDF], fit_name: str, verbose=True) -> pd.DataFrame:
    columns = ['Datnum', 'ESC /mV', 'Theta_cold /mV (real)', 'Theta_hot /mV (real)', 'dT /mV (real)']
    datas = []
    for dat in dats:
        fit_cold, fit_hot = [dat.SquareEntropy.get_fit(fit_name=fit_name+'_'+temp) for temp in ['cold', 'hot']]
        tcold, thot = [fit.best_values.theta for fit in [fit_cold, fit_hot]]
        dT = thot-tcold
        datas.append([dat.datnum, dat.Logs.fds['ESC'], tcold/1000, thot/1000, dT/1000])

    df = pd.DataFrame(data=datas, columns=columns)
    df = df.set_index('Datnum')
    if verbose:
        print(df.to_markdown())
    return df


def linear_fit_dTs(x, dTs) -> FitInfo:
    line = lm.models.LinearModel()
    fit = line.fit(dTs, x=x)
    fit = FitInfo.from_fit(fit)
    return fit





TRANSITION_DATNUMS = list(range(1604, 1635, 2))
TRANSITION_DATNUMS_2 = list(range(1833, 1866, 2))  # Taken at ESS = -340mV

DATNUMS1 = list(range(1637, 1652 + 1))  # First set all the way from weakly coupled to gamma broadened
DATNUMS2 = list(range(1653, 1668 + 1))  # Second set all the way from weakly coupled to gamma broadened
DATNUMS3 = list(range(1669, 1684 + 1))  # Second set all the way from weakly coupled to gamma broadened

VS_TIME = list(range(1685, 1772 + 1))

POS2 = list(range(1778, 1794 + 1))  # First set at different ESS (-350mV instead of -375mV)
POS3 = list(range(1798, 1814 + 1))  # ESS at -340mV
POS3_2 = list(range(1815, 1831 + 1))  # ESS at -340mV
POS3_3 = list(range(1869, 1918 + 1, 2))  # ESS at -340mV (alternates with Transition Only)
POS3_3.remove(1909)

POS3_3_Tonly = list(range(1870, 1918 + 1, 2))  # Same as above but transition only scans
POS3_3_Tonly.remove(1910)

POS3_100 = list(range(1919, 1986 + 1, 2))
POS3_100_Tonly = list(range(1920, 1986 + 1, 2))

CONST_GAMMA = list(range(1995, 2003 + 1, 2))
CONST_GAMMA_Tonly = list(range(1996, 2004 + 1, 2))

CONST_GAMMA_2 = list(range(2005, 2014 + 1, 2))
CONST_GAMMA_Tonly_2 = list(range(2006, 2014 + 1, 2))

POS4 = list(range(2015, 2064 + 1, 2))
POS4_Tonly = list(range(2016, 2064 + 1, 2))

VS_HEATER = list(range(2082, 2089 + 1, 2))
VS_HEATER_Tonly = list(range(2083, 2089 + 1, 2))

LONG = list(range(2089, 2094 + 1, 2))
LONG_Tonly = list(range(2090, 2094 + 1, 2))

POS2_2 = list(range(2095, 2134 + 1, 2))
POS2_2Tonly = list(range(2096, 2134 + 1, 2))

LONG2 = list(range(2164, 2172 + 1, 3))
LONG2_Tonly = list(range(2165, 2172 + 1, 3))
LONG2_CSQs = list(range(2173, 2175 + 1))  # retaken afterwards to be off transition

IDTransition1 = list(range(2383, 2408 + 1, 2))
IDTransition1_Tonly = list(range(2383, 2408 + 1, 2))

IDTransition2 = list(range(2409, 2434 + 1, 2))
IDTransition2_Tonly = list(range(2410, 2434 + 1, 2))

IDTransition3 = list(range(2435, 2454 + 1, 2))
IDTransition3_Tonly = list(range(2436, 2454 + 1, 2))

Tonly = list(range(2499, 2506 + 1))

IDP1 = list(range(2601, 2658 + 1, 2))
IDP1_Tonly = list(range(2602, 2658 + 1, 2))

IDPtemp = list(range(2661, 2798 + 1, 2))
IDPtemp_Tonly = list(range(2662, 2798 + 1, 2))

ID_mar18 = list(range(2815, 2975 + 1))
ID_mar18.remove(2849)

ID_mar18_wide = list(range(2976, 3016 + 1))

# At 100mK... Dot tunes are 3063, 3064
ID_virtual1 = list(range(3066, 3080 + 1))  # 100mK, wrong direction in IP1
ID_virtual2 = list(range(3085, 3244 + 1)) + list(range(3430, 3450 + 1))  # 100mK
ID_normal2 = list(range(3245, 3429 + 1))  # 100mK

# At 50mK... Dot tunes are 3653 and 3654
ID_virtual3 = list(range(3451, 3551 + 1))  # 50mK
ID_normal3 = list(range(3552, 3652 + 1))  # 50mK

# LONG_GAMMA = [2164, 2167, 2170, 2213, 2216]
# LONG_GAMMA_Tonly = [2165, 2168, 2171, 2214, 2217]
# LONG_GAMMA_csq = [2173, 2174, 2175, 2215, 2218]
LONG_GAMMA = [2164, 2167, 2170, 2216]
LONG_GAMMA_Tonly = [2165, 2168, 2171, 2217]
LONG_GAMMA_csq = [2173, 2174, 2175, 2218]

GAMMA_DCbias = list(range(2219, 2230 + 1, 1))  # Alternates +/- bias

CD2_Tonly = list(range(5303, 5317 + 1))
CD2_Tonly2 = list(range(5322, 5326 + 1))
CD2_Tonly3 = list(range(5328, 5348 + 1))

QUICK_70_HEAT = list(range(6457, 6468 + 1))
LONG_30_HEAT = list(range(6469, 6496 + 1, 2))
LONG_30_HEAT_Tonly = list(range(6470, 6496 + 1, 2))

LONG_30_HEAT2 = list(range(6501, 6532 + 1, 2))
LONG_30_HEAT_Tonly2 = list(range(6502, 6532 + 1, 2))

MULTIPLE_30_HEAT = list(range(6469, 6500+1, 2)) + list(range(6501, 6532+1, 2)) + list(range(6551, 6550+1, 2)) + \
                   list(range(6551, 6654+1, 2))#+ list(range(6655, 6686+1, 2))
MULTIPLE_30_HEAT_Tonly = list(range(6469+1, 6500+1, 2)) + list(range(6501+1, 6532+1, 2)) + list(range(6551+1, 6550+1, 2)) + \
                   list(range(6551+1, 6654+1, 2))# + list(range(6655+1, 6686+1, 2))
[MULTIPLE_30_HEAT.remove(v) for v in [6561, 6587, 6593, 6613, 6643, 6617]]
[MULTIPLE_30_HEAT_Tonly.remove(v+1) for v in [6561, 6587, 6593, 6613, 6643, 6617]]

MORE_SYMMETRIC = list(range(6715, 6746 + 1, 2))
MORE_SYMMETRIC_Tonly = list(range(6716, 6746 + 1, 2))

MORE_SYMMETRIC_LONG = list(range(6747, 6772 + 1, 2)) + list(range(6777, 6790+1, 2))
MORE_SYMMETRIC_LONG_Tonly = list(range(6748, 6772 + 1, 2)) + list(range(6778, 6790+1, 2))


MORE_SYMMETRIC_LONG2 = list(range(7322, 7357 + 1, 2))
MORE_SYMMETRIC_LONG_Tonly2 = list(range(7323, 7357 + 1, 2))

if __name__ == '__main__':
    # entropy_datnums = ID_normal3
    entropy_datnums = MORE_SYMMETRIC_LONG2
    # entropy_datnums = ID_virtual3
    transition_datnums = MORE_SYMMETRIC_LONG_Tonly2

    # entropy_datnums = LONG_GAMMA
    # transition_datnums = LONG_GAMMA_Tonly
    # csq_datnums = LONG_GAMMA_csq
    csq_datnums = []

    # Which things to plot
    plot_transition_fitting = False
    plot_transition_values = True
    plot_entropy_vs_gamma = True
    plot_entropy_vs_time = False
    plot_amp_comparison = False
    plot_csq_map_check = False
    plot_stacked_square_heated = False
    plot_stacked_transition = False
    plot_dot_tune = False
    print_info = False

    # For resetting dats if something goes badly wrong
    reset_dats = False
    if reset_dats:
        for datnum in transition_datnums + entropy_datnums + csq_datnums:
            get_dat(datnum, overwrite=True)

    # Calculations
    gamma_scans = True
    if gamma_scans:
        csq_map = False
        calculate = True
        overwrite = False
        theta = None
        gamma = 0
        width = None
        dt_from_self = True
        # dt = 1.111
        dt = 1.45*9.423
        amp = None
        x_func = lambda dat: dat.Logs.fds['ESC']
        x_label = 'ESC /mV'
        t_func_name = 'i_sense_digamma'
        save_name = 'gamma'
        # ess = lambda dat: dat.Logs.fds['ESS']
        ess = lambda dat: dat.Logs.bds['ESS']
        # title_append = lambda dats: f'CSS={dats[0].Logs.bds["CSS"]:.1f}mV'
        title_append = lambda dats: f' at ESS={ess(dats[0]):.1f}mV, CSS={dats[0].Logs.bds["CSS"]:.1f}mV'
        sub_linear_entropy = True
        sub_lin_width = lambda dat: abs(dat.Data.x[-1] - dat.Data.x[0]) / 6
        # centering_threshold = -30  # Below this in ESC will be centered
        centering_threshold = -180  # Below this in ESC will be centered
    else:
        csq_map = False
        calculate = True
        overwrite = False
        theta = 4.2
        gamma = None
        width = 200
        dt_from_self = False
        dt = 0.947  # 100mK 3nA
        # dt = 0.8  # 50mK 2nA
        amp = 0.405
        # x_func = lambda dat: dat.Logs.fds['IP1/200']
        # x_label = 'IP1/200 /mV'
        x_func = lambda dat: dat.Logs.fds['ESC']
        x_label = 'ESC /mV'
        # x_func = lambda dat: np.mean(dat.Data.sweepgates_x[1][1:])
        # x_label = 'IP1*200 /mV'
        t_func_name = 'i_sense_digamma_amplin'
        # t_func_name = 'i_sense'
        save_name = 'digamma_amplin'
        # save_name = 'normal_isense'
        ess = lambda dat: dat.Logs.bds['ESS']
        # title_append = lambda dats: f'CSS={dats[0].Logs.bds["CSS"]:.1f}mV'
        title_append = lambda dats: f' at ESS={ess(dats[0]):.1f}mV'
        sub_linear_entropy = True
        sub_lin_width = lambda dat: abs(dat.Data.x[-1] - dat.Data.x[0]) / 3
        centering_threshold = -30  # Below this in ESC will be centered

    if calculate:
        with ProcessPoolExecutor() as pool:
            if csq_map:
                # for num in [1619]:
                for num in csq_datnums:
                    setup_csq_dat(num, overwrite=overwrite)

                print(f'Done Setting up CSQonly dats')

                # # # TESTING
                # calculate_csq_map(transition_datnums[0], csq_datnum=None, overwrite=False)
                # calculate_csq_map(entropy_datnums[0], csq_datnum=None, overwrite=False)

                # Do CSQ mapping for regular data for all dats
                for ds, csq_dat in progressbar(
                        zip([(ed, td) for ed, td in zip(entropy_datnums, transition_datnums)], csq_datnums)):
                    list(pool.map(partial(calculate_csq_map, csq_datnum=csq_dat, overwrite=overwrite),
                                  ds))
                print(f'Done CSQ Mapping')

            # TESTING
            # do_transition_only_calc(datnum=transition_datnums[0], save_name=save_name,
            #                         theta=theta, gamma=gamma, t_func_name=t_func_name,
            #                         csq_mapped=csq_map, overwrite=overwrite)
            # do_entropy_calc(datnum=entropy_datnums[0], save_name=save_name,
            #                 setpoint_start=0.005, t_func_name='i_sense',
            #                 theta=theta, gamma=gamma, width=width,
            #                 csq_mapped=csq_map, overwrite=overwrite)
            # set_sf_from_transition([entropy_datnums[0]], [transition_datnums[0]], fit_name=save_name,
            #                        integration_info_name=save_name)
            # print('Starting Transition_only')

            # Transition Only
            list(pool.map(partial(do_transition_only_calc, save_name=save_name,
                                  theta=theta, gamma=gamma, t_func_name=t_func_name,
                                  csq_mapped=csq_map,
                                  centering_threshold=centering_threshold,
                                  overwrite=overwrite),
                          transition_datnums))
            print(f'Done Fitting Transition Only')

            # Entropy Only
            list(pool.map(partial(do_entropy_calc, save_name=save_name,
                                  setpoint_start=0.005, t_func_name='i_sense',
                                  theta=theta, gamma=gamma, width=width,
                                  csq_mapped=csq_map,
                                  center_for_avg=centering_threshold,
                                  overwrite=overwrite), entropy_datnums))
            print(f'Done Fitting Entropy')

            if amp and dt:
                for num in progressbar(entropy_datnums):
                    dat = get_dat(num)
                    dat.Entropy.set_integration_info(dT=dt,
                                                     amp=amp if amp is not None else np.nan,
                                                     name=save_name,
                                                     overwrite=True)
            else:
                set_sf_from_transition(entropy_datnums, transition_datnums, fit_name=save_name,
                                       integration_info_name=save_name,
                                       dt_from_self=dt_from_self,
                                       fixed_dt=dt, fixed_amp=amp)

            print(f'Done setting Integrated Info')

    if plot_transition_fitting:
        dat = get_dat(2168)  # Can do 2064 for less data
        csq_dat = 2174
        fit_name = 'test'
        # fit_name = 'narrow'
        fit_func = 'i_sense_digamma'
        # fit_func = 'i_sense_digamma'
        transition_only = False
        csq_map = True
        theta = 3.9
        gamma = None
        fit_width = 600

        for fit_width in [300, 500, 700, 1000]:
            # for csq_map in [False, True]:
            title_row_2 = f'<br>Func={fit_func}'
            if csq_map:
                title_row_2 = f'{title_row_2}. Mapped to CSQ using dat{csq_dat}'
            y_label = f'CSGate /mV' if csq_map else f'Current /nA'
            y_units = 'mV' if csq_map else 'nA'

            if transition_only:
                fit = do_transition_only_calc(dat.datnum, save_name=fit_name,
                                              theta=theta, gamma=gamma, width=fit_width, t_func_name=fit_func,
                                              csq_mapped=csq_map, overwrite=False)
            else:
                raise NotImplementedError

            # do_narrow_fits([dat], theta=3.9756, gamma=None, width=fit_width, output_name='SPS.005', overwrite=True,
            #                fit_func=fit_func, fit_name=fit_name, transition_only=True)
            plotter = OneD(dat=dat)
            fig_fit = plotter.figure(
                title=f'Dat{dat.datnum}: Transition Data with Fit (width={fit_width})' + title_row_2,
                ylabel=y_label)
            fig_fit.add_trace(
                single_transition_trace(dat, label='Data', fit_name=fit_name, transition_only=transition_only,
                                        csq_mapped=csq_map, se_output_name=save_name))
            fig_fit.add_trace(single_transition_trace(dat, label='Fit', fit_only=True, fit_name=fit_name,
                                                      transition_only=transition_only, se_output_name=save_name,
                                                      csq_mapped=csq_map))

            fig_minus = plotter.figure(
                title=f'Dat{dat.datnum}: Transition Data minus Fit (width={fit_width})' + title_row_2,
                ylabel=f'{C.DELTA}{y_label}')
            fig_minus.add_trace(single_transition_trace(dat, label=None, subtract_fit=True, fit_name=fit_name,
                                                        transition_only=transition_only, se_output_name=save_name,
                                                        csq_mapped=csq_map))

            for fig in [fig_minus, fig_fit]:
                plotter.add_line(fig, value=fit_width, mode='vertical')
                plotter.add_line(fig, value=-fit_width, mode='vertical')

            if fit_func == 'i_sense_digamma_amplin':
                add_print = f'\tGamma: {fit.best_values.g:.1f}mV\n' + \
                            f'\tAmpLin: {fit.best_values.get("amplin", 0):.3g}{y_units}/mV\n'
            elif fit_func == 'i_sense_digamma':
                add_print = f'\tGamma: {fit.best_values.g:.1f}mV\n'
            else:
                add_print = ''

            print(f'Dat{dat.datnum}:\n'
                  f'\tWidth: {C.PM}{fit_width}mV\n'
                  f'\tAmp: {fit.best_values.amp:.3f}{y_units}\n'
                  f'\tTheta: {fit.best_values.theta:.2f}mV\n'
                  f'\tLin: {fit.best_values.lin:.3g}{y_units}/mV\n'
                  f'\tConst: {fit.best_values.const:.1f}{y_units}\n'
                  f'\tCenter: {fit.best_values.mid:.1f}mV\n' + add_print
                  )

            fig_fit.show()
            fig_minus.show()

    if plot_transition_values:
        transition_only = True
        fit_name = save_name
        param = 'theta'
        if transition_only:
            all_dats = get_dats(transition_datnums)
            fig = transition_fig(dats=all_dats, xlabel='ESC /mV', title_append=' vs ESC for Transition Only scans',
                                 param=param)
            for dnums, label in zip([transition_datnums], ['Set 1', 'Set 2']):
                all_dats = get_dats(dnums)
                fig.add_trace(transition_trace(all_dats, x_func=x_func, from_square_entropy=False,
                                               fit_name=save_name, param=param, label=label))
                print(
                    f'Avg weakly coupled cold theta = '
                    f'{np.mean([dat.Transition.get_fit(name=fit_name).best_values.theta for dat in all_dats if dat.Logs.fds["ESC"] <= -330])}')
        else:
            all_dats = get_dats(entropy_datnums)
            fig = transition_fig(dats=all_dats, xlabel='ESC /mV', title_append=' vs ESC for Entropy scans', param=param)
            for datnums, label in zip([entropy_datnums], ['Set 1', 'Set 2']):
                all_dats = get_dats(datnums)
                fig.add_trace(transition_trace(all_dats, x_func=x_func, from_square_entropy=True,
                                               fit_name=save_name, param=param, label=label))
                print(
                    f'Avg weakly coupled cold theta = {np.mean([dat.SquareEntropy.get_fit(which_fit="transition", fit_name=fit_name).best_values.theta for dat in all_dats if dat.Logs.fds["ESC"] <= -330])}')
        fig.show()

    if plot_entropy_vs_gamma:
        integration_info_name = save_name
        all_dats = get_dats(entropy_datnums)
        fig = get_integrated_fig(all_dats, title_append=title_append(all_dats))

        all_dats = get_dats(entropy_datnums)
        fig.add_trace(get_integrated_trace(dats=all_dats, x_func=x_func, x_label=x_label,
                                           trace_name='Set 1',
                                           save_name=save_name,
                                           int_info_name=integration_info_name, SE_output_name=save_name,
                                           sub_linear=sub_linear_entropy, signal_width=sub_lin_width))

        # fig2 = plot_fit_integrated_comparison(dats, x_func=x_func, x_label=x_label,
        #                                       int_info_name=integration_info_name, fit_name=save_name,
        #                                       plot=True)
        fig.show()

        all_dats = get_dats(entropy_datnums)
        # dats = [dat for dat in dats if np.isclose(dat.Logs.bds['CSS'], -25, atol=1)]
        all_dats = [dat for dat in all_dats if 0 < np.nanmean(dat_integrated_sub_lin(dat, signal_width=sub_lin_width(dat), int_info_name=save_name)[-50:]) < 2]
        fig = get_integrated_fig(all_dats, title_append=title_append(all_dats))
        dat_chunks = [[dat for dat in all_dats if np.isclose(dat.Logs.bds['CSS'], css, atol=1)] for css in set([dat.Logs.bds['CSS'] for dat in all_dats])]
        # dat_chunks = [[dat for dat in dats if np.isclose(dat.Logs.bds['ESS'], css, atol=1)] for css in set([dat.Logs.bds['ESS'] for dat in dats])]
        for all_dats in dat_chunks:
            n = f'CSS={all_dats[0].Logs.bds["CSS"]:.1f}mV'
            # n = f'ESS={dats[0].Logs.bds["ESS"]:.1f}mV'
            fig.add_trace(get_integrated_trace(dats=all_dats, x_func=x_func, x_label=x_label,
                                               trace_name=n,
                                               save_name=save_name,
                                               int_info_name=integration_info_name, SE_output_name=save_name,
                                               sub_linear=sub_linear_entropy, signal_width=sub_lin_width))
        fig.show()

    if plot_amp_comparison:
        compare_amps = True
        compare_integrated = True

        entropy_dats = get_dats(entropy_datnums)
        transition_dats = get_dats(transition_datnums)

        if compare_amps:
            fig = transition_fig(entropy_dats + transition_dats, xlabel='ESC /mV',
                                 title_append=' Comparison of amp from Entropy vs Transition only', param='amp')
            fig.add_trace(transition_trace(entropy_dats, x_func=x_func,
                                           from_square_entropy=True, fit_name=save_name, param='amp',
                                           label='Cold part of Entropy',
                                           mode='markers+lines'))
            fig.add_trace(transition_trace(transition_dats, x_func=x_func,
                                           from_square_entropy=False, fit_name=save_name, param='amp',
                                           label='Transition Only',
                                           mode='markers+lines'))
            fig.show()
        if compare_integrated:
            fig = get_integrated_fig(title_append=f' at ESS = {entropy_dats[0].Logs.fds["ESS"]}mV<br>'
                                                  f'Comparison of amp from Entropy vs Transition')
            for integration_info_name, label in zip(['first', 'amp from transition'],
                                                    ['Amp from Entropy', 'Amp from Transition']):
                fig.add_trace(get_integrated_trace(dats=entropy_dats, x_func=x_func,
                                                   trace_name=label,
                                                   save_name=save_name,
                                                   int_info_name=integration_info_name, SE_output_name=save_name))
            fig.show()

    if plot_entropy_vs_time:
        all_dats = get_dats(VS_TIME)
        fit_fig = entropy_vs_time_fig(title=f'Dats{all_dats[0].datnum}-{all_dats[-1].datnum}: Fit Entropy vs Time')
        int_fig = entropy_vs_time_fig(title=f'Dats{all_dats[0].datnum}-{all_dats[-1].datnum}: Integrated Entropy vs Time')
        for esc in set([dat.Logs.fds['ESC'] for dat in all_dats]):
            ds = [dat for dat in all_dats if dat.Logs.fds['ESC'] == esc]
            fit_fig.add_trace(entropy_vs_time_trace(dats=ds, integrated=False, trace_name=f'Fit Entropy {esc}mV',
                                                    fit_name=save_name, integrated_name='first'))
            int_fig.add_trace(entropy_vs_time_trace(dats=ds, integrated=True, trace_name=f'Integrated Entropy {esc}mV',
                                                    fit_name=save_name, integrated_name='first'))

        fit_fig.show()
        int_fig.show()

    if plot_csq_map_check:
        # dat = get_dat(1619)
        dat = get_dat(2174)
        csq_dat = 2174
        plotter = OneD(dat=dat)
        calculate_csq_map(dat.datnum, csq_datnum=csq_dat, overwrite=True)

        fig = plotter.figure(xlabel=dat.Logs.xlabel, ylabel='Current /nA',
                             title=f'Dat{dat.datnum}: Before mapping i_sense to CSQ<br>'
                                   f'ESS={dat.Logs.fds["ESS"]:.1f}mV, '
                                   f'ESC={dat.Logs.fds["ESC"]:.1f}mV, '
                                   f'ESP={dat.Logs.fds["ESP"]:.1f}mV'
                             )
        x = dat.Data.get_data('x')
        data = dat.Data.get_data('i_sense')
        fig.add_trace(plotter.trace(x=x, data=data, mode='lines'))
        fig.show()

        fig = plotter.figure(xlabel=dat.Logs.xlabel, ylabel='Gate /mV',
                             title=f'Dat{dat.datnum}: After mapping i_sense to CSQ (using Dat{csq_dat})<br>'
                                   f'ESS={dat.Logs.fds["ESS"]:.1f}mV, '
                                   f'ESC={dat.Logs.fds["ESC"]:.1f}mV, '
                                   f'ESP={dat.Logs.fds["ESP"]:.1f}mV'
                             )
        data = dat.Data.get_data('csq_mapped')
        fig.add_trace(plotter.trace(x=x, data=data, mode='lines'))
        fig.show()

        line = lm.models.LinearModel()
        params = line.make_params()
        params = U.edit_params(params, ['slope', 'intercept'], [1, 200])
        fit = line.fit(data=data.astype(np.float32), x=x.astype(np.float32), params=params, nan_policy='omit')
        fig = plotter.figure(xlabel=dat.Logs.xlabel, ylabel=f'{C.DELTA}Gate /mV',
                             title=f'Dat{dat.datnum}: After mapping and subtracting line fit (using Dat{csq_dat})<br>'
                                   f'ESS={dat.Logs.fds["ESS"]:.1f}mV, '
                                   f'ESC={dat.Logs.fds["ESC"]:.1f}mV, '
                                   f'ESP={dat.Logs.fds["ESP"]:.1f}mV'
                             )
        fig.add_trace(plotter.trace(x=x, data=data - fit.eval(x=x)))
        fig.show()

    if plot_dot_tune:
        dat = get_dat(3063)
        plotter = TwoD(dat=dat)
        fig = plotter.plot(data=dat.Data.i_sense, title=f'Dat{dat.datnum}: Dot Tune')
        fig.show()

        data = dat.Data.i_sense
        diff_data = np.diff(dat_analysis.hdf_util.T, axis=0)
        # x = U.get_matching_x(dat.Data.x, diff_data)
        x = dat.Data.x
        y = U.get_matching_x(dat.Data.y, shape_to_match=diff_data.shape[0])
        fig = plotter.plot(diff_data, x=x, y=y, title=f'Dat{dat.datnum}: Differentiated Dot Tune',
                           trace_kwargs=dict(zmin=-0.1, zmax=np.nanmax(diff_data)))
        fig.show()

    if print_info:
        all_dats = get_dats(entropy_datnums)
        for dat in all_dats:
            int_ds = np.nanmean(dat.Entropy.get_integrated_entropy(name=save_name, data=dat.SquareEntropy.get_Outputs(
                name=save_name).average_entropy_signal)[-10:])
            int_info = dat.Entropy.get_integration_info(name=save_name)
            print(f'Dat{dat.datnum}:\n'
                  # f'\tIP/200={dat.Logs.fds["IP1/200"]:.0f}mV\n'
                  f'\tESC={dat.Logs.fds["ESC"]:.0f}mV\n'
                  f'\tESP={dat.Logs.fds["ESP"]:.1f}mV\n'
                  f'\tFit dS={dat.Entropy.get_fit(name=save_name).best_values.dS:.3f}kB\n'
                  f'\tInt dS={int_ds:.3f}kB\n'
                  f'\tdT = {int_info.dT:.3f}mV\n'
                  f'\tamp = {int_info.amp:.3f}nA\n'
                  f'\tsf = {int_info.sf:.2f}\n')

        all_dats = get_dats(transition_datnums)
        for dat in all_dats:
            print(f'Dat{dat.datnum}:\n'
                  # f'\tIP/200={dat.Logs.fds["IP1/200"]:.0f}mV\n'
                  f'\tESC={dat.Logs.fds["ESC"]:.0f}mV\n'
                  f'\tESP={dat.Logs.fds["ESP"]:.1f}mV\n'
                  f'\tGamma/kT = {dat.Transition.get_fit(name=save_name).best_values.g / theta:.1f}\n')

        # dts = {}
        # for dat in dats:
        #     fit_c = dat.SquareEntropy.get_fit(which_fit='transition', transition_part='cold', fit_name=save_name+'_cold')
        #     fit_h = dat.SquareEntropy.get_fit(which_fit='transition', transition_part='hot',
        #                                       fit_name=save_name+'_hot')
        #     if all([fit.best_values.theta is not None for fit in [fit_c, fit_h]]):
        #         print(f'Dat{dat.datnum}:\nHot theta = {fit_h.best_values.theta:.4f}mV\n'
        #               f'Cold theta = {fit_c.best_values.theta:.4f}mV\n'
        #               f'dT = {fit_h.best_values.theta-fit_c.best_values.theta:.4f}mV')
        #         dts[dat.datnum] = fit_h.best_values.theta-fit_c.best_values.theta
        #     else:
        #         pass
        #         # print(f'Dat{dat.datnum}: \nhot = {fit_h},\ncold = {fit_c}')
