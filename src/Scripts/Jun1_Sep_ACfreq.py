from src.Scripts.StandardImports import *
from src.Configs import Sep19Config, Mar19Config
from src.DatCode import Entropy as E


if __name__ == '__main__':
    jan_datdf = IDD.get_exp_df('jan20', dfname='May20')
    sep_datdf = IDD.get_exp_df('sep19', dfname='May20')
    mar_datdf = IDD.get_exp_df('mar19', dfname='May20')

    dats = [C.DatHandler.get_dat(num, 'base', sep_datdf, Sep19Config) for num in range(1049, 1055 + 1)]
    for dat in dats:
        params = E.get_param_estimates(dat.Entropy.x_array, dat.Entropy._data, dat.Transition.fit_values.mids,
                                       dat.Transition.fit_values.thetas)
        params = [CU.edit_params(param, ['const', 'theta', 'mid'],
                                 [0, None, None], [False, True, True],
                                 max_val=[None, 500, None]) for param in params]
        params = CU.ensure_params_list(params, dat.Entropy._data)
        dat.Entropy.recalculate_fits(params=params)
        DF.update_save(dat, update=True, save=False, datdf=sep_datdf)
    sep_datdf.save()

    fig, ax = plt.subplots(1)
    ax.cla()
    for dat in dats:
        x = dat.Data.x_array
        x, data = CU.sub_poly_from_data(x, dat.Transition._avg_data, dat.Transition._avg_full_fit)

        PF.display_1d(x, data, ax,
                      label=f'{dat.datnum}, {dat.Instruments.srs1.freq:.1f}, {dat.Logs.time_elapsed / len(dat.Logs.y_array):.1f}, {dat.Entropy.dS:.2f}')
    PF.ax_setup(ax, 'Avg I_sense varying freq and delay', dat.Logs.x_label, dat.Logs.y_label, legend=True)
    ax.legend().set_title('Datnum, Freq, Duration, dS')