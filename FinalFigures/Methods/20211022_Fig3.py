from dat_analysis.dat_object.make_dat import get_dat, get_dats
from dat_analysis.plotting.plotly.dat_plotting import OneD


def current_dat():
    dat = get_dat(1155)
    cold = dat.SquareEntropy.get_fit(which='avg', calculate_only=True, transition_part='cold', output_name = 'SPS.0045')
    hot = dat.SquareEntropy.get_fit(which='avg', calculate_only=True, transition_part='hot', output_name = 'SPS.0045')
    dt = hot.best_values.theta - cold.best_values.theta
    dat.Entropy.set_integration_info(dt, cold.best_values.amp, name='SPS.0045')
    ii = dat.Entropy.get_integration_info('SPS.0045')
    integrated = ii.integrate(dat.SquareEntropy.get_Outputs(name='SPS.0045').average_entropy_signal)
    plotter = OneD(dat)
    fig = plotter.plot(integrated, x=dat.SquareEntropy.avg_x, ylabel='Entropy /kB', title=f'Dat{dat.datnum}: Integrated entropy using dT = {dt:.3f} after ignoring 4.5ms per step')
    fig.show()

def hot_vs_cold_new_dat():
    dat = get_dat(2164)
    fit_name = 'SPS.01'
    transition_cold = dat.SquareEntropy.get_transition_part(name=fit_name, part='cold')
    transition_hot = dat.SquareEntropy.get_transition_part(name=fit_name, part='hot')
    x = dat.SquareEntropy.avg_x

    plotter = OneD(dat)
    fig = plotter.plot(transition_cold, x=x, trace_name='Cold', mode='lines')
    fig.add_trace(plotter.trace(transition_hot, x=x, name='Hot', mode='lines'))

    fig.show()


if __name__ == '__main__':
    dat = get_dat(1155)
    name = 'SPS.0045'
    entropy = dat.SquareEntropy.get_Outputs(name=name).average_entropy_signal
    x = dat.SquareEntropy.avg_x
    plotter = OneD(dat)
    fig = plotter.plot(entropy, x=x, trace_name='Data', mode='lines', title=f'Dat{dat.datnum}')
    fig.add_trace(
    plotter.trace(dat.Entropy.get_fit(name=name).eval_fit(x), x=x, name='fit', mode='lines'))
    fig.show()