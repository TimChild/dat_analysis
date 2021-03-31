from src.DatObject.Make_Dat import get_dat
from src.Dash.DatPlotting import OneD
import src.UsefulFunctions as U
from src.DatObject.Attributes.Transition import i_sense, i_sense_digamma, i_sense_digamma_amplin
from src.AnalysisTools.fitting import _get_data_in_range

import lmfit as lm
import numpy as np
import plotly.io as pio

pio.renderers.default = "browser"

if __name__ == '__main__':
    dat = get_dat(2214)
    x = dat.Data.get_data('x')
    data = dat.Data.get_data('i_sense')[63]

    plotter = OneD(dat=dat)
    plotter.MAX_POINTS = 100000
    fig = plotter.figure(ylabel='Current /nA', title=f'Dat{dat.datnum}: Checking Accuracy of Center from fit')

    # Whole row of data
    fig.add_trace(plotter.trace(x=x, data=data, name='All data of row0', mode='lines'))

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
            temp_x, temp_data = _get_data_in_range(x, data, width=w, center=0)
            fit = dat.Transition.get_fit(x=temp_x, data=temp_data, calculate_only=True, fit_func=func,
                                         initial_params=params)
            fits.append(fit)
            mid = fit.best_values.mid
            fig.add_trace(
                plotter.trace(x=x, data=fit.eval_fit(x=x), name=f'Data +-{w} around transition<br>{name}: {mid:.2f}',
                              mode='lines'))
            reports.append(f'{name} fit to +-{w}mV of data around transition with all params vary:\n{fit.fit_report}')

    for fit in fits:
        f = fit.fit_result
        pars = f.params
        for p in pars:
            pars[p].stderr = 0.1 * pars[p].value

        f.conf_interval()

    for fit in fits:
        f = fit.fit_result
        print(lm.printfuncs.ci_report(f.ci_out))
    fig.show()

    fig.write_html('figs/centering_accuracy.html')


    all_data = dat.Transition.data
    x_500, _ = _get_data_in_range(x, all_data[0], width=500)
    data_500 = np.array([_get_data_in_range(x, d, width=500)[1] for d in all_data])

    fits = [dat.Transition.get_fit(which='row', row=i, name='amplin_500', initial_params=params, fit_func=i_sense_digamma_amplin,
                                   data=d, x=x_500, check_exists=False) for i, d in enumerate(data_500)]

    fig2 = plotter.figure(xlabel='Center from fit /mV', ylabel='Data Row', title=f'Dat{dat.datnum}: Variation of '
                                                                                 f'center fit value over time')
    # centers = [f.best_values.mid for f in dat.Transition.get_row_fits(name='amplin_500')]
    centers = []
    ys = []
    for i in range(data_500.shape[0]):
        try:
            centers.append(dat.Transition.get_fit(which='row', row=i, name='amplin_500'))
            ys.append(i)
        except U.NotFoundInHdfError:
            print(f'Failed to find on row {i}')
    centers = [c.best_values.mid for c in centers]


    # fig2.add_trace(plotter.trace(x=centers, data=dat.Data.get_data('y'), mode='markers'))
    fig2.add_trace(plotter.trace(x=centers, data=ys, mode='markers', trace_kwargs=dict(markers=dict(size=3))))
    fig2.show()
