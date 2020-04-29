from src.Scripts.StandardImports import *
import lmfit as lm
import src.DatCode.Transition as T
from scipy.special import digamma


dats = make_dats([1516, 1522, 1525, 1557, 1560, 1563, 1566, 1533], dfname='Apr20', datname='digamma')
dat = dats[0]
fig, ax = plt.subplots(1)
x = dat.Data.x_array
dat.Transition.recalculate_fits(func=T.i_sense_digamma)
ax.plot(dat.Data.x_array, dat.Transition._avg_data)
ax.plot(dat.Transition.avg_x_array, dat.Transition._avg_full_fit.best_fit, c='C3', label='digamma_fit')
df1 = CU.fit_info_to_df([dat.Transition._avg_full_fit])
df1['index'] = 'digamma'
dat.Transition.recalculate_fits(func=T.i_sense_digamma_quad)
df2 = CU.fit_info_to_df([dat.Transition._avg_full_fit])
df2['index'] = 'digamma w/ quad'
ax.plot(dat.Transition.avg_x_array, dat.Transition._avg_full_fit.best_fit, c='C4', label='digamma_quad_fit')
df = df1.append(df2, sort=True)
PF.plot_df_table(df, f'digamma fit without vs with quadratic term for dat[{dat.datnum}]')




# def i_sense_digamma(x, mid, g, theta, amp, lin, const):
#     arg = digamma(0.5 + (x-mid + 1j * g / 2) / (2 * np.pi * 1j * theta))  # j is imaginary i
#     return amp * (0.5 + np.imag(arg) / np.pi) + lin * (x-mid) + const - amp/2  # +amp/2 so const term coincides with i_sense


isens = lm.models.Model(T.i_sense)
idg = lm.models.Model(T.i_sense_digamma)
params = lm.Parameters()
params.add_many(('mid', 10, True, None, None, None, None),
                        ('theta', 3, True, 0, None, None, None),
                        ('amp', 4, True, 0, None, None, None),
                        ('lin', 0.01, True, 0, None, None, None),
                        ('const', 2, True, None, None, None, None),
                        ('g', 0, True, None, None, None, None))
fig, ax = plt.subplots(1)
x = np.linspace(-100, 100, 1000)
ax.plot(x, isens.eval(params=params, x=x), label='i_sense')
ax.plot(x, idg.eval(params=params, x=x), label='di_gamma')
ax.legend()