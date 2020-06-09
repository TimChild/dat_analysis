from src.Scripts.StandardImports import *
from src.DatCode import Transition as T, DCbias as DC

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
    sdcdat = C.make_dat_standard(1945, 'base', 'overwrite', dattypes={'transition', 'dcbias'}, datdf=sep_datdf,
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


def _get_quad_min(quad_pars):
    quad = lm.models.QuadraticModel()
    a = quad_pars['a'].value
    b = quad_pars['b'].value
    c = quad_pars['c'].value
    return quad.eval(quad_pars, x=(-b / (2 * a)))


def _get_shift(i_heat, dc_mid_poly_pars):
    quad = lm.models.QuadraticModel()
    mid = quad.eval(dc_mid_poly_pars, x=0)
    return quad.eval(dc_mid_poly_pars, x=i_heat) - mid


def _get_dt(i_heat, dc_theta_poly_pars):
    quad = lm.models.QuadraticModel()
    theta_min = _get_quad_min(dc_theta_poly_pars)
    theta_hot = np.average([quad.eval(dc_theta_poly_pars, x=heat) for heat in [i_heat, -i_heat]])
    return theta_hot - theta_min


def _get_isense(x, i_pars_zero, dc_mid_poly_pars, dc_theta_poly_pars, i_heat):
    model = lm.models.Model(T.i_sense_digamma)
    shift = _get_shift(i_heat, dc_mid_poly_pars)
    dt = _get_dt(i_heat, dc_theta_poly_pars)
    theta_0 = i_pars_zero['theta'].value
    mid = i_pars_zero['mid'].value
    i_pars = CU.edit_params(i_pars_zero, ['theta', 'mid'], [theta_0 + dt, mid + shift], [False, False])
    return model.eval(i_pars, x=x)


# TODO: just need to make a set of I_sense params with theta set at 0 heating (i.e. theta fit - dt/2).
#  Then I can pass in model params for how the middle shifts, and how to caluclate dt and I will get back an array
#  which I can plot several of for +iheat -iheat and 0heat, then .... I think this replicates Silvia's work


def _get_dc_mid_fit(dc_biases, dc_mids, with_lin=True):
    x = dc_biases
    z = dc_mids
    quad = lm.models.QuadraticModel()
    quad.guess(z, x=x)  # get starting values
    pars = quad.guess(z, x=x)
    quad_fit = quad.fit(z, params=pars, x=x, nan_policy='omit')
    if with_lin is False:
        pars = CU.edit_params(pars, 'b', 0, False)
        quad_fit = quad.fit(z, params=pars, x=x, nan_policy='omit')

    return quad_fit


if __name__ == '__main__':
    # j1dats = [C.DatHandler.get_dat(num, 'base', jan_datdf, config=None) for num in jan1_datnums]
    j1_IDDs = [IDD.InDepthData(num, set_name='jan20_gamma', run_fits=False, show_plots=False, datdf=jan_datdf) for num
               in jan1_datnums]
    # j2_IDDs = [IDD.InDepthData(num, set_name='jan20_gamma_2', run_fits=False, show_plots=False, datdf=jan_datdf) for num in jan2_datnums]
    s_IDDs = [IDD.InDepthData(num, set_name='sep19_gamma', run_fits=False, show_plots=False, datdf=sep_datdf) for num in
              sep_datnums]
    m_IDDs = [IDD.InDepthData(num, set_name='mar19_gamma_entropy', run_fits=False, show_plots=False, datdf=mar_datdf) for
              num in mar_datnums]
    mdcs = list(range(3385, 3410 + 1, 2))
    mdcs = [C.DatHandler.get_dat(num, 'base', mar_datdf, Mar19Config) for num in mdcs]


    show_extra_plots = False
    march_data = False
    dc_mid_lin_force_zero = False
    fig, axs = PF.make_axes(4)

    idd = j1_IDDs[1]
    # Set cut off values for DC_bias mid data
    dc_cutoff_vals = (-185, 190)

    # idd = j2_IDDs[0]
    # Set cut off values for DC_bias mid data
    # dc_cutoff_vals = (-185, 190)

    # idd = s_IDDs[0]
    # Set cut off values for DC_bias mid data
    # dc_cutoff_vals = (None, None)

    # idd = m_IDDs[0]
    # Set cut off values for DC_bias mid data
    # dc_cutoff_vals = (None, None)

    quad = lm.models.QuadraticModel()
    i_pars: lm.Parameters
    i_pars = idd.setup_meta.dat.Transition._avg_full_fit.params
    i_pars.add('g', 0, False, -20, np.inf)
    idd.run_all_fits(i_params=i_pars)  # Check that fits actually worked after this

    print(f'Numerical dIsense for dat{idd.datnum}')

    i_heat = idd.setup_meta.dat.Instruments.srs1.out / 50 * np.sqrt(2)
    if march_data is False:
        dc = idd.setup_meta.dc  # type: make_dat_standard()
        dt = dc.DCbias.get_dt_at_current(i_heat)
    else:
        dc = None
        dt = IDD.get_mar19_dt(i_heat)
    print(f'Using dT = {dt}')

    if None not in dc_cutoff_vals:
        print(f'Using cutoff values [{dc_cutoff_vals}] for DCbias mid data')
        ids = CU.get_data_index(dc.Data.y_array, dc_cutoff_vals)  # To avoid and beginning/end of data
    else:
        ids = (None, None)

    if march_data is False:
        dc_biases = np.array(dc.Data.y_array[ids[0]:ids[1]]) / 10
        dc_mids = np.array(dc.Transition.fit_values.mids[ids[0]:ids[1]]).astype(np.float32)
        dc_theta_quad_pars = dc.DCbias.full_fit.params
    else:
        dc_biases = np.array([dat.Logs.dacs[0]/10 for dat in mdcs]).astype(np.float32)
        dc_mids = np.array([dat.Transition.avg_fit_values.mids[0] for dat in mdcs]).astype(np.float32)
        dc_thetas = np.array([dat.Transition.avg_fit_values.thetas[0] for dat in mdcs]).astype(np.float32)
        dc_theta_quad_pars = quad.fit(dc_thetas, quad.guess(dc_thetas, x=dc_biases), x=dc_biases).params
    if show_extra_plots is True:  # To see mid/theta data being fitted to
        _, ax = PF.make_axes(1)
        ax[0].plot(dc_biases, dc_mids, label='DCbias mid data')
        if march_data is False:
            dc.DCbias.plot_self(dc, idd.setup_meta.dat)

    dc_mid_fit = _get_dc_mid_fit(dc_biases, dc_mids, with_lin=True)
    dc_mid_quad_pars = dc_mid_fit.params
    if dc_mid_lin_force_zero is True:
        dc_mid_quad_pars['b'].value = 0

    ax = axs[0]
    ax.cla()
    idd.Plot.plot_avg_i(idd, ax, centered=True, sub_const=True, sub_lin=True, sub_quad=True)
    PF.ax_text(ax, f'Amp={idd.i_avg_fit.amp:.3f}', loc=(0.05, 0.05))
    PF.ax_setup(ax, 'Initial Data (for amp)')
    if show_extra_plots is True:
        df = CU.fit_info_to_df([idd.i_avg_fit.fit], uncertainties=True)
        PF.plot_df_table(df, f'Dat{idd.datnum} fit values')

    ax = axs[1]
    ax.cla()
    if march_data is False:
        min_theta = _get_quad_min(dc.DCbias.full_fit.params)
    else:
        min_theta = np.nanmin(quad.eval(dc_theta_quad_pars, x=np.linspace(-5, 5, 500)))
    i_pars_zero = CU.edit_params(idd.i_avg_fit.params, ['theta', 'mid', 'lin', 'const'], [min_theta, 0, 0, 0],
                                 [False, False, False, False])

    x = idd.i_avg_fit.x - idd.i_avg_fit.mid
    i_cold = _get_isense(x, i_pars_zero, dc_mid_quad_pars, dc_theta_quad_pars, 0)
    i_plus, i_minus = [_get_isense(x, i_pars_zero, dc_mid_quad_pars, dc_theta_quad_pars, ih) for ih in
                       [i_heat, -i_heat]]

    ax.plot(x, i_cold, label=f'Ih=0, {Char.THETA}={min_theta:.2f}', color='blue')
    ax.plot(x, i_plus, label=f'Ih={i_heat:.1f}nA, {Char.THETA}={min_theta + dt:.2f}', color='red')
    ax.plot(x, i_minus, label=f'Ih={-i_heat:.1f}nA, {Char.THETA}={min_theta + dt:.2f}', color='orange')
    PF.ax_setup(ax, f'Centered Transition sub poly\nw/ I_heat = 0, +{i_heat:.1f}nA, {-i_heat:.1f}nA',
                'Plunger (offset) /mV', 'Current (offset) /nA', legend=True)

    ax = axs[2]
    ax.cla()
    ent = ((i_cold - i_plus) + (i_cold - i_minus)) / 2 / 2
    ax.plot(x, ent, color='C3', label='Calculated Entropy')
    ax.scatter(idd.x - idd.i_avg_fit.mid, idd.e_avg, s=1)
    PF.add_scatter_label(label='Data', ax=ax)
    e_pars = idd.e_avg_fit.params
    e_pars = CU.edit_params(e_pars, 'mid', 0)
    e_fit = idd.e_avg_fit.fit.model.fit(ent, e_pars, x=x)
    PF.ax_text(ax, f'dS_calc={e_fit.best_values["dS"]:.3f}, dS_data={idd.e_avg_fit.dS:.3f}', loc=(0.05, 0.05))
    PF.ax_setup(ax, 'Calculated Entropy Signal /2\nRaw data *sqrt(2)', 'Plunger (offset) /mV', 'Entropy signal /nA',
                legend=True)

    ax = axs[3]
    ax.cla()
    ax.scatter(dc_biases, dc_mids, label='Data', s=1)
    ax.plot(dc_biases, quad.eval(dc_mid_quad_pars, x=dc_biases), color='C3', label='Fit')
    # ax.plot(dc_biases, poly_lin_fit.best_fit, color='C5', label='Lin forced 0')
    if show_extra_plots is True:
        dc_mid_df = CU.fit_info_to_df([dc_mid_fit], uncertainties=True, sf=3)
        PF.plot_df_table(dc_mid_df, 'Poly fit through DCbias mids')
    a, b, c = [CU.sig_fig(dc_mid_quad_pars[key].value, 3) for key in dc_mid_quad_pars.keys()]
    c = dc_mid_quad_pars['c'].value
    PF.ax_text(ax, f'{a:.2f}x^2+{b:.2f}x+{c:.1f}', loc=(0.5, 0.05))
    PF.ax_setup(ax, 'DCbias mid values with fit', 'I_heat /nA', 'Transition mid /mV', legend=True)

    view_width = 15
    # view_width = 150
    for ax in axs[0:3]:
        ax.set_xlim(-view_width, +view_width)
    fig.suptitle(f'{idd.setup_meta.dat.config_name[0:5]}-Dat{idd.datnum}: Numerical dIsense')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

