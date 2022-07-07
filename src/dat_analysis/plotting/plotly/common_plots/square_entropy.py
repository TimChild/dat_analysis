from typing import Optional, TYPE_CHECKING

import dat_analysis.analysis_tools.transition
from dat_analysis.dat_object import dat_hdf
from dat_analysis.dat_object.attributes.SquareEntropy import entropy_signal
from dat_analysis.plotting.plotly import OneD, TwoD

if TYPE_CHECKING:
    from dat_analysis.analysis_tools.general_fitting import FitInfo
    import plotly.graph_objects as go


class SquareEntropyPlotter:
    def __init__(self, dat: dat_hdf):
        self.dat: dat_hdf = dat
        self.one_plotter: OneD = OneD(dat)
        self.two_plotter: TwoD = TwoD(dat)

    def plot_raw_row(self, row: Optional[int] = None) -> go.Figure:
        """Raw data by row"""
        if row is None:
            row = 0
        z = dat_analysis.analysis_tools.transition.i_sense
        z = z[row]

        fig = self.one_plotter.plot(z, mode='lines', title=f'Dat{self.dat.datnum}: Row {row} of I_sense')
        return fig

    def plot_transition_row(self, row: Optional[int] = None) -> go.Figure:
        """Single row of data after averaging setpoints and cycles"""
        if row is None:
            row = 0
        z = self.dat.SquareEntropy.default_Output.cycled
        x = self.dat.SquareEntropy.x
        z = z[row]

        fig = self.one_plotter.figure(title=f'Dat{self.dat.datnum}: Row {row} of cycled')
        for data, label in zip(z, ['v0_0', 'vP', 'v0_1', 'vM']):
            fig.add_trace(self.one_plotter.trace(data, name=label, x=x, mode='lines'))

        return fig

    def plot_transition_avg(self) -> go.Figure:
        """Centered and averaged I_sense, not yet entropy signal"""
        z = self.dat.SquareEntropy.default_Output.averaged
        x = self.dat.SquareEntropy.x
        fig = self.one_plotter.figure(title=f'Dat{self.dat.datnum}: Centered and Averaged I_sense')

        for row, label in zip(z, ['v0_0', 'vP', 'v0_1', 'vM']):
            fig.add_trace(self.one_plotter.trace(row, name=label, x=x, mode='lines'))
        return fig

    def plot_entropy_avg(self) -> go.Figure:
        """Averaged Entropy signal"""
        z = self.dat.SquareEntropy.avg_entropy_signal
        x = self.dat.SquareEntropy.x

        fit_info = self.dat.Entropy.avg_fit
        fit_x = self.dat.Entropy.avg_x
        fit = fit_info.eval_fit(fit_x)

        fig = self.one_plotter.figure(title=f'Dat{self.dat.datnum}: Average Entropy Signal')

        fig.add_trace(self.one_plotter.trace(data=z, x=x, mode='lines', name='Entropy Signal'))
        fig.add_trace(self.one_plotter.trace(data=fit, x=fit_x, mode='lines', name='Fit'))

        self.one_plotter.add_textbox(fig, text=f'Fit Values:\n'
                                               f'dS={fit_info.best_values.dS:.3f}',
                                     position='TR')
        return fig

    def plot_entropy_row(self, row: Optional[int] = None) -> go.Figure:
        """Single row of entropy signal"""
        if row is None:
            row = 0
        z = self.dat.SquareEntropy.default_Output.cycled
        x = self.dat.SquareEntropy.x
        z = z[row]
        z = entropy_signal(z)

        fig = self.one_plotter.figure(title=f'Dat{self.dat.datnum}: Row {row} Entropy Signal')
        fig.add_trace(self.one_plotter.trace(data=z, x=x, mode='markers', name=f'Row {row} data'))

        fit: FitInfo = self.dat.Entropy.row_fits[row]

        fig.add_trace(self.one_plotter.trace(data=fit.eval_fit(x=x), x=x, mode='lines', name='Fit'))
        return fig

    def plot_integrated_entropy_avg(self) -> go.Figure:
        """Averaged Entropy signal"""
        z = self.dat.Entropy.integrated_entropy
        x = self.dat.Entropy.avg_x

        fig = self.one_plotter.figure(title=f'Dat{self.dat.datnum}: Average Integrated Entropy')
        fig.add_trace(self.one_plotter.trace(data=z, x=x, mode='lines'))
        return fig
