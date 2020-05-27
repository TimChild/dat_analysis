from src.Scripts.StandardImports import *

from src.Configs import Sep19Config, Mar19Config

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
# j1_IDDs = [IDD.InDepthData(num, set_name='jan20_gamma', run_fits=False, show_plots=False, datdf=jan_datdf) for num in jan1_datnums]
# j2_IDDs = [IDD.InDepthData(num, set_name='jan20_gamma_2', run_fits=False, show_plots=False, datdf=jan_datdf) for num in jan2_datnums]
s_IDDs = [IDD.InDepthData(num, set_name='sep19_gamma', run_fits=False, show_plots=False, datdf=sep_datdf) for num in
          sep_datnums]
m_IDDs = [IDD.InDepthData(num, set_name='mar19_gamma_entropy', run_fits=False, show_plots=False, datdf=mar_datdf) for
          num in mar_datnums]

# idd = j1_IDDs[0]
