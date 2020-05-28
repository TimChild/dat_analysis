from src.Scripts.StandardImports import *

from src.Configs import Sep19Config, Mar19Config
import lmfit as lm
import copy

jan_datdf = IDD.get_exp_df('jan20', dfname='May20')
sep_datdf = IDD.get_exp_df('sep19', dfname='May20')
mar_datdf = IDD.get_exp_df('mar19', dfname='May20')

jan1_datnums = IDD.InDepthData.get_datnums('jan20_gamma')
jan2_datnums = IDD.InDepthData.get_datnums('jan20_gamma_2')
sep_datnums = IDD.InDepthData.get_datnums('sep19_gamma')
mar_datnums = IDD.InDepthData.get_datnums('mar19_gamma_entropy')


def _make_initial_dats():
    j1dats = [C.make_dat_standard(num, 'base', 'overwrite', datdf=jan_datdf) for num in jan1_datnums]
    j2dats = [C.make_dat_standard(num, 'base', 'overwrite', datdf=jan_datdf) for num in jan2_datnums]
    jdcdat = C.make_dat_standard(1529, 'base', 'overwrite', datdf=jan_datdf)

    for dat in j1dats + j2dats + [jdcdat]:
        DF.update_save(dat, update=True, save=False, datdf=jan_datdf)
    jan_datdf.save()

    sdats = [C.make_dat_standard(num, 'base', 'overwrite', dattypes={'entropy'}, datdf=sep_datdf, config=Sep19Config)
             for num in sep_datnums]
    sdcdat = C.make_dat_standard(1945, 'base', 'overwrite', dattypes={'transition'}, datdf=sep_datdf,
                                 config=Sep19Config)
    mdats = [C.make_dat_standard(num, 'base', 'overwrite', dattypes={'entropy'}, datdf=mar_datdf, config=Mar19Config)
             for num in mar_datnums]
    mdcdats = [
        C.make_dat_standard(num, 'base', 'overwrite', dattypes={'transition'}, datdf=mar_datdf, config=Mar19Config) for
        num in list(range(3385, 3410 + 1, 2))]

    for dat in sdats + [sdcdat]:
        DF.update_save(dat, update=True, save=False, datdf=sep_datdf)
    sep_datdf.save()

    for dat in mdats + mdcdats:
        DF.update_save(dat, update=True, save=False, datdf=mar_datdf)
    mar_datdf.save()


# j1dats = [C.DatHandler.get_dat(num, 'base', jan_datdf, config=None) for num in jan1_datnums]
j1_IDDs = [IDD.InDepthData(num, set_name='jan20_gamma', run_fits=False, show_plots=False, datdf=jan_datdf) for num in jan1_datnums]
# j2_IDDs = [IDD.InDepthData(num, set_name='jan20_gamma_2', run_fits=False, show_plots=False, datdf=jan_datdf) for num in jan2_datnums]
# s_IDDs = [IDD.InDepthData(num, set_name='sep19_gamma', run_fits=False, show_plots=False, datdf=sep_datdf) for num in
#           sep_datnums]
# m_IDDs = [IDD.InDepthData(num, set_name='mar19_gamma_entropy', run_fits=False, show_plots=False, datdf=mar_datdf) for
#           num in mar_datnums]


if __name__ == '__main__':
    idd = j1_IDDs[0]

    i_pars: lm.Parameters
    i_pars = idd.setup_meta.dat.Transition._avg_full_fit.params
    i_pars.add('g', 0, False, -20, np.inf)
    idd.run_all_fits(i_params=i_pars)  # Check that fits actually worked

    print(f'Numerical dIsense for dat{idd.datnum}')
    dc = idd.setup_meta.dc  # type: make_dat_standard()
    i_heat = idd.setup_meta.dat.Instruments.srs1.out/50*np.sqrt(2)
    dt = dc.DCbias.get_dt_at_current(i_heat)
    print(f'Using dT = {dt}')

    fig, axs = PF.make_axes(6)

    ax = axs[0]
    ax.cla()
    idd.Plot.plot_avg_i(idd, ax, centered=True, sub_const=True, sub_lin=True, sub_quad=True)
    PF.ax_setup(ax, 'Initial Data (for amp)')
    df = CU.fit_info_to_df([idd.i_avg_fit.fit], uncertainties=True)
    PF.plot_df_table(df, f'Dat{idd.datnum} fit values')

    ax = axs[1]
    ax.cla()
    i_pars_cold = copy.deepcopy(idd.i_avg_fit.params)
    i_pars_hot = copy.deepcopy(idd.i_avg_fit.params)
    i_pars_cold = CU.edit_params(i_pars_cold, 'theta', idd.i_avg_fit.theta - dt / 2, False)
    i_pars_hot = CU.edit_params(i_pars_hot, 'theta', idd.i_avg_fit.theta + dt / 2, False)
    i_model = idd.i_avg_fit.fit.model
    x = idd.i_avg_fit.x - idd.i_avg_fit.mid
    i_pars_cold = CU.edit_params(i_pars_cold, ['const', 'lin', 'mid', 'quad'], [0, 0, 0, 0],
                                 [False, False, False, False])
    i_pars_hot = CU.edit_params(i_pars_hot, ['const', 'lin', 'mid', 'quad'], [0, 0, 0, 0],
                                [False, False, False, False])
    i_cold = i_model.eval(x=x, params=i_pars_cold)
    i_hot = i_model.eval(x=x, params=i_pars_hot)

    ax.plot(x, i_cold, label=f'Cold ({idd.i_avg_fit.theta - dt/2})', color='blue')
    ax.plot(x, i_hot, label=f'Hot ({idd.i_avg_fit.theta + dt/2})', color='red')
    PF.ax_setup(ax, 'Same amp, theta +- abs dT/2\nall other params 0', 'Plunger (offset) /mV', 'Current (offset) /nA')

    ax = axs[2]
    ax.cla()
    ent = i_cold - i_hot
    ax.plot(x, ent, label='no shift')
    PF.ax_setup(ax, 'Calculated Entropy Signal', 'Plunger (offset) /mV', 'Entropy signal /nA', legend = True)

    ax = axs[3]
    ax.cla()
    poly_model = lm.models.QuadraticModel()
    cutoff_vals = (-185, 190)
    if cutoff_vals is not None:
        print(f'Using cutoff values [{cutoff_vals}] for DCbias mid data')
        ids = CU.get_data_index(dc.Data.y_array, cutoff_vals)  # To avoid and beginning/end of data
    else:
        ids = (None, None)
    x_dc = np.array(dc.Data.y_array[ids[0]:ids[1]]) / 10
    z = np.array(dc.Transition.fit_values.mids[ids[0]:ids[1]]).astype(np.float32)
    ax.plot(x_dc, z, label='DCbias mid data')
    poly_model.guess(z, x=x_dc)  # get starting values
    pars = poly_model.guess(z, x=x_dc)
    pars_lin = copy.deepcopy(pars)
    pars_lin = CU.edit_params(pars_lin, 'b', 0, False)
    poly_fit = poly_model.fit(z, params=pars, x=x_dc, nan_policy='omit')
    poly_lin_fit = poly_model.fit(z, params=pars_lin, x=x_dc, nan_policy='omit')

    ax.plot(x_dc, poly_fit.best_fit, color='C3', label='Full quad fit')
    ax.plot(x_dc, poly_lin_fit.best_fit, color='C5', label='Lin forced 0')

    polydf = CU.fit_info_to_df([poly_fit], uncertainties=True, sf=3)
    PF.plot_df_table(polydf, 'Poly fit through DCbias mids')
    poly_lindf = CU.fit_info_to_df([poly_lin_fit], uncertainties=True, sf=3)
    PF.plot_df_table(poly_lindf, 'Poly fit through DCbias mids - Lin forced zero')

    PF.ax_setup(ax, 'DCbias mid values with fit', 'I_heat /nA', 'Transition mid /mV', legend=True)

    ax = axs[4]
    ax.cla()
    # poly_lin_a = poly_lin_fit.best_values['a']
    # poly_lin_c = poly_lin_fit.best_values['c']
    # shift = np.mean([poly_lin_fit.eval(x=heat) for heat in [i_heat]]) - poly_lin_c

    # poly_a = poly_fit.best_values['a']
    # poly_b = poly_fit.best_values['b']
    # poly_c = poly_fit.best_values['c']
    # shift = poly_fit.eval(x=-i_heat) - poly_c

    shift = -0.12

    i_pars_hot = CU.edit_params(i_pars_hot, ['mid'], [shift])
    i_hot = i_model.eval(x=x, params=i_pars_hot)
    ax.plot(x, i_cold, label=f'Cold ({idd.i_avg_fit.theta - dt / 2:.2f})', color='blue')
    ax.plot(x, i_hot, label=f'Hot ({idd.i_avg_fit.theta + dt / 2:.2f})', color='red')
    PF.ax_setup(ax, f'Same amp, theta +- abs dT/2\nhot shifted by {shift:.3f}mV',
                'Plunger (offset) /mV', 'Current (offset) /nA', legend=True)

    # poly_a = poly_fit.best_values['a']
    # poly_b = poly_fit.best_values['b']
    # shift = poly_a*i_heat**2 + poly_b*i_heat

    ax = axs[5]
    ax.cla()
    ent = np.asarray(i_cold - i_hot)
    ax.plot(x, ent, label='with shift', color='C3')

    ax.scatter(idd.x-idd.e_avg_fit.mid, idd.e_avg, s=1)
    PF.add_scatter_label(label='Data', ax=ax)
    PF.ax_setup(ax, 'Calculated Entropy Signal /2rt2', 'Plunger (offset) /mV', 'Entropy signal /nA', legend=True)

    e_pars = idd.e_avg_fit.params
    e_pars = CU.edit_params(e_pars, 'mid', 0)
    e_fit = idd.e_avg_fit.fit.model.fit(ent, e_pars, x=x)
    PF.ax_text(ax, f'dS={e_fit.best_values["dS"]:.3f}', loc=(0.05, 0.05))
