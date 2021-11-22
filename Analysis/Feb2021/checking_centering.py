"""
Sep 21 -- I think this is a fairly comprehensive comparison of whether centering data first before averaging is actually
a good thing to do. For noisy data, the fits to individual rows of data may introduce more error than the real shift
of charge transition

Not useful enough as is to be worth extracting. Better off to remake this analysis and plots if/when I need to do it
again
"""

from dat_analysis.dat_object.make_dat import get_dat
from dat_analysis.plotting.plotly.dat_plotting import OneD
import dat_analysis.useful_functions as U
from dat_analysis.dat_object.attributes.Transition import i_sense, i_sense_digamma, i_sense_digamma_amplin
from dat_analysis.analysis_tools.general_fitting import get_data_in_range

import numpy as np
import plotly.io as pio
import plotly.express as px

pio.renderers.default = "browser"


if __name__ == '__main__':
    # dat = get_dat(2214)
    # x = dat.Data.get_data('x')
    # data = dat.Data.get_data('i_sense')[63]
    # all_data = dat.Transition.data

    # dat = get_dat(2216)
    dat = get_dat(2164)
    out = dat.SquareEntropy.get_row_only_output(name='default')
    x = out.x
    all_data = np.nanmean(np.array(out.cycled[:, (0, 2), :]), axis=1)
    single_row = 10
    data = all_data[single_row]

    plotter = OneD(dat=dat)
    plotter.MAX_POINTS = 100000
    fig = plotter.figure(ylabel='Current /nA', title=f'Dat{dat.datnum}: Checking Accuracy of Center from fit')

    # Whole row of data
    fig.add_trace(plotter.trace(x=x, data=data, name=f'All data of row{single_row}', mode='lines'))

    # Fits
    reports = []
    fits = []
    params = dat.Transition.get_default_params(x, data)
    params.add('g', value=0, min=-50, max=1000)
    params.add('amplin', value=0)
    params['theta'].vary = False
    params['theta'].value = 4

    for func, name in zip([i_sense, i_sense_digamma], ['i_sense', 'i_sense_digamma']):
        fit = dat.Transition.get_fit(x=x, data=data, calculate_only=True, fit_func=func, initial_params=params)
        fits.append(fit)
        mid = fit.best_values.mid
        fig.add_trace(plotter.trace(x=x, data=fit.eval_fit(x=x), name=f'{name} - all data: {mid:.2f}', mode='lines'))
        reports.append(f'{name} fit to all data with all params vary:\n{fit.fit_report}')

    # Binned data
    data_binned, x_binned = U.resample_data(data, x, max_num_pnts=100)
    fig.add_trace(plotter.trace(x=x_binned, data=data_binned, mode='lines', name='Binned'))

    for s, name in zip([0, 1], ['even', 'odd']):
        temp_x = x_binned[s::2]
        temp_data = data_binned[s::2]
        fit = dat.Transition.get_fit(x=temp_x, data=temp_data, calculate_only=True, fit_func=i_sense,
                                     initial_params=params)
        fits.append(fit)
        mid = fit.best_values.mid
        fig.add_trace(plotter.trace(x=x, data=fit.eval_fit(x=x), name=f'{name} binned data: {mid:.2f}', mode='lines'))
        reports.append(f'i_sense fit to {name} binned data with all params vary:\n{fit.fit_report}')

    # Fit widths
    for func, name in zip([i_sense, i_sense_digamma, i_sense_digamma_amplin],
                          ['i_sense', 'i_sense_digamma', 'i_sense_digamma_amplin']):
        for w in [500, 1000, 2000]:
            temp_x, temp_data = get_data_in_range(x, data, width=w, center=0)
            fit = dat.Transition.get_fit(x=temp_x, data=temp_data, calculate_only=True, fit_func=func,
                                         initial_params=params)
            fits.append(fit)
            mid = fit.best_values.mid
            fig.add_trace(
                plotter.trace(x=x, data=fit.eval_fit(x=x), name=f'Data +-{w} around transition<br>{name}: {mid:.2f}',
                              mode='lines'))
            reports.append(f'{name} fit to +-{w}mV of data around transition with all params vary:\n{fit.fit_report}')

    # for fit in fits:
    #     f = fit.fit_result
    #     pars = f.params
    #     for p in pars:
    #         pars[p].stderr = 0.1 * pars[p].value
    #
    #     f.conf_interval()
    #
    # for fit in fits:
    #     f = fit.fit_result
    #     print(lm.printfuncs.ci_report(f.ci_out))
    fig.show()

    fig.write_html('figs/centering_accuracy.html')

    x_500, _ = get_data_in_range(x, all_data[0], width=500)
    data_500 = np.array([get_data_in_range(x, d, width=500)[1] for d in all_data])

    fits = [dat.Transition.get_fit(which='row', row=i, name='amplin_500', initial_params=params, fit_func=i_sense_digamma_amplin,
                                   data=d, x=x_500, check_exists=False) for i, d in enumerate(data_500)]

    fig2 = plotter.figure(xlabel='Center from fit /mV', ylabel='Data Row', title=f'Dat{dat.datnum}: Variation of '
                                                                                 f'center fit value over time')
    # centers = [f.best_values.mid for f in dat.Transition.get_row_fits(name='amplin_500')]
    fits = []
    ys = []
    for i in range(data_500.shape[0]):
        try:
            fits.append(dat.Transition.get_fit(which='row', row=i, name='amplin_500'))
            ys.append(i)
        except U.NotFoundInHdfError:
            print(f'Failed to find on row {i}')
    centers = [c.best_values.mid for c in fits]


    # fig2.add_trace(plotter.trace(x=centers, data=dat.Data.get_data('y'), mode='markers'))
    fig2.add_trace(plotter.trace(x=centers, data=ys, mode='markers+lines', trace_kwargs=dict(marker=dict(size=3))))
    fig2.show()

    fig3 = px.histogram(x=centers, nbins=200)
    fig3.update_layout(title=f'Dat{dat.datnum}: Histogram of Centers', xaxis_title='Center in ACC*100 /mV')
    fig3.show()

    near_zeros = []
    near_fifteens = []
    others = []
    zero_rows = []
    not_zero_rows = []
    for i, fit in enumerate(fits):
        mid = fit.params['mid']
        if abs(mid.value) < 2:
            near_zeros.append(fit)
            zero_rows.append(i)
        elif abs(mid.value - 15) < 2:
            near_fifteens.append(fit)
            not_zero_rows.append(i)
        else:
            others.append(fit)
            not_zero_rows.append(i)

    par = 'g'
    for fs in [near_zeros, near_fifteens, others]:
        print(np.mean([f.best_values.get(par) for f in fs if f.best_values.get(par) is not None]))


    x = [f.best_values.mid for f in fits if f.best_values.mid is not None]
    z = [f.params['mid'].stderr for f in fits if f.best_values.mid is not None]

    fig = plotter.plot(data=z, x=x, xlabel='Center in ACC*100 /mV', ylabel='Uncertainty in Fit value /mV',
                       title=f'Dat{dat.datnum}: Correlation of Center to fit value uncertainty',
                       trace_kwargs=dict(marker=dict(size=3)))
    fig.show()

    z = [f.reduced_chi_sq for f in fits if f.best_values.mid is not None]
    fig = plotter.plot(data=z, x=x, xlabel='Center in ACC*100 /mV', ylabel='Reduced Chi square of Fit',
                       title=f'Dat{dat.datnum}: Correlation of Center to Reduced Chi squaure of Fit',
                       trace_kwargs=dict(marker=dict(size=3)))
    fig.show()


    for rows, name in zip([zero_rows, not_zero_rows], ['zero only', 'NOT zero']):
        entropy_data = np.nanmean(dat.SquareEntropy.get_row_only_output(name='forced_theta_linear').entropy_signal[rows], axis=0)
        int_info = dat.Entropy.get_integration_info('scaled_dT')
        integrated = int_info.integrate(entropy_data)
        fig = plotter.plot(data=integrated, x=out.x, title=f'Dat{dat.datnum}: Integrated Entropy of rows where center is {name}',
                           mode='lines', ylabel='Entropy /kB')
        fig.show()

