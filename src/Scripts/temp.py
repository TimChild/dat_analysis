# from src.DataStandardize.ExpSpecific.Aug20 import add_temp_magy
from src.Scripts.StandardImports import *
from src.Plotting.Plotly.PlotlyUtil import PlotlyViewer as PV
from progressbar import progressbar
import plotly.graph_objects as go
from src.DataStandardize.ExpSpecific.Sep20 import Fixes
from dataclasses import dataclass


if __name__ == '__main__':
    dat = get_dat(1304)



if __name__ == '__main_':
    dats = get_dats((1304,
                     1312 + 1))  # Same as above but now with HQPC biases right, Much faster than above so probably more noisy

    # dats = get_dats((1359, 1360+1))  # Scan along short part of transition 40mV/s 5 cycles, -202mT
    for dat in dats:
        Fixes._add_magy(dat)

    save_graphs = False
    recalculate = True
    for dat in progressbar(dats):
        if not hasattr(dat.Other, 'time_processed') or recalculate is True:
            bin_data = True
            num_per_row = 400

            sub_line = False

            allowed_amps = (0.8, 1.2)
            default_amp = 1.05
            allowed_dTs = (1, 15)
            default_dT = 5.96

            ys = dat.SquareEntropy.y
            z = dat.SquareEntropy.Processed.outputs.cycled

            xs, trans_datas, entropy_datas, integrated_datas = list(), list(), list(), list()
            tcs, ths, dTs, mids, amps, sfs = list(), list(), list(), list(), list(), list()
            fit_entropies, integrated_entropies = list(), list()

            for trans_data in z:
                x = dat.SquareEntropy.Processed.outputs.x
                entropy_data = SE.entropy_signal(trans_data)

                data_tcs, data_amps, data_centers = list(), list(), list()
                for data in trans_data[0::2]:
                    fit = T.transition_fits(x, data, func=T.i_sense)[0]
                    data_tcs.append(fit.best_values['theta'])
                    data_amps.append(fit.best_values['amp'])
                    data_centers.append(fit.best_values['mid'])

                data_ths = list()
                for data in trans_data[1::2]:
                    fit = T.transition_fits(x, data, func=T.i_sense)[0]
                    data_ths.append(fit.best_values['theta'])

                tc = np.nanmean(data_tcs)
                th = np.nanmean(data_ths)
                dT = th - tc

                amp = np.nanmean(data_amps)
                mid = np.average(data_centers)

                mid = mid if (-1000 < mid < 1000) else 0
                if not (allowed_amps[0] < amp < allowed_amps[1]):
                    amp = default_amp
                if not (allowed_dTs[0] < dT < allowed_dTs[1]):
                    dT = default_dT
                dx = np.mean(np.diff(x))
                sf = SE.scaling(dt=dT, amplitude=amp, dx=dx)
                int_info = SE.IntegratedInfo(dT=dT, amp=amp, dx=dx)
                integrated_data = SE.integrate_entropy(entropy_data, int_info.sf)

                if sub_line is True:
                    line = lm.models.LinearModel()
                    indexs = CU.get_data_index(x, [mid - 2000, mid - 400])
                    line_fit = line.fit(integrated_data[indexs[0]:indexs[1]], x=x[indexs[0]:indexs[1]],
                                        nan_policy='omit')
                    integrated_data = integrated_data - line_fit.eval(x=x)

                indexs = CU.get_data_index(x, [mid + 300, mid + 1000])
                int_dS = np.mean(integrated_data[indexs[0]:indexs[1]])

                # Calculate Nik Fit
                e_pars = E.get_param_estimates(x, entropy_data)[0]
                e_pars = CU.edit_params(e_pars, 'const', 0, True)
                e_pars = CU.edit_params(e_pars, 'dS', min_val=0, max_val=2)
                efit = E.entropy_fits(x, entropy_data, params=e_pars)[0]
                efit_info = DA.FitInfo()
                efit_info.init_from_fit(efit)

                if bin_data is True:
                    bin_size = np.ceil(x.shape[-1] / num_per_row)
                    trans_data = CU.bin_data(trans_data, bin_size)
                    entropy_data = CU.bin_data(entropy_data, bin_size)
                    integrated_data = CU.bin_data(integrated_data, bin_size)
                    x = np.linspace(x[0], x[-1], int(x.shape[-1] / bin_size))

                tcs.append(tc)
                ths.append(th)
                dTs.append(dT)
                sfs.append(sf)
                xs.append(x)
                amps.append(amp)
                mids.append(mid)
                trans_datas.append(trans_data)
                entropy_datas.append([efit_info.eval_fit(x=x), entropy_data])
                integrated_datas.append(integrated_data)
                integrated_entropies.append(int_dS)
                fit_entropies.append(efit_info.best_values.dS)

            dat.Other.tcs = tcs
            dat.Other.ths = ths
            dat.Other.dTs = dTs
            dat.Other.amps = amps
            dat.Other.mids = mids
            dat.Other.sfs = sfs

            dat.Other.set_data('xs', np.asanyarray(xs))
            dat.Other.ys = ys[:]
            dat.Other.set_data('trans_datas', np.asanyarray(trans_datas))
            dat.Other.set_data('entropy_datas', np.asanyarray(entropy_datas))
            dat.Other.set_data('integrated_datas', np.asanyarray(integrated_datas))

            dat.Other.integrated_entropies = integrated_entropies
            dat.Other.fit_entropies = fit_entropies
            dat.Other.time_processed = str(pd.Timestamp.now())
            dat.Other.update_HDF()








    # dats = get_dats(range(337, 400))
    # dat = dats[0]
    #
    # add_temp_magy(dats)
    #
    # datas, xs, ids, titles = list(), list(), list(), list()
    # traces = list()
    # for dat in dats:
    #     dat.Other.magy: float
    #     data = CU.decimate(dat.Transition.avg_data, dat.Logs.Fastdac.measure_freq, 30)
    #     x = np.linspace(dat.Data.x_array[0], dat.Data.x_array[-1], data.shape[-1])
    #     x, data = CU.sub_poly_from_data(x, data, dat.Transition.avg_fit)
    #     datas.append(data)
    #     xs.append(x)
    #     ids.append(dat.datnum)
    #     name = f'Dat{dat.datnum}: Field={dat.Other.magy:.0f}mT, Bias={dat.Logs.fds["R2T(10M)"]:.0f}mV'
    #     titles.append(name)
    #     traces.append(go.Scatter(x=x, y=data, mode='lines', name=name))
    # xlabel = 'LP*2/mV'
    # ylabel = 'Current /nA'
    #
    # # fig = PL.get_figure(datas, xs, ids=ids, titles=titles, xlabel=xlabel, ylabel=ylabel)
    #
    # fig = go.Figure()
    # fig.add_traces(traces)
    #
    # v = PV(fig)
    #
    #
    # P.display_2d()

    # get_dat(85)