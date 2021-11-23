from typing import Iterable, Optional

import lmfit as lm
import numpy as np
from plotly import graph_objs as go, graph_objects as go


def theta_vs_fridge_temp_fig(thetas: Iterable[float], temps: Iterable[float], datnums: Iterable[int],
                             lower_fit_temp_limit: Optional[float] = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(mode='markers+lines+text', x=temps, y=thetas, text=datnums, name='Avg Thetas'))

    if lower_fit_temp_limit is not None:
        line = lm.models.LinearModel()
        fit_thetas, fit_temps = np.array(
            [(theta, temp) for theta, temp in zip(thetas, temps) if temp > lower_fit_temp_limit]).T
        fit = line.fit(fit_thetas, x=fit_temps, nan_policy='omit')
        x_range = np.linspace(0, max(temps), int(max(temps)) + 1)
        fig.add_trace(go.Scatter(mode='lines', x=x_range, y=fit.eval(x=x_range),
                                 name=f'Fit to temps > {lower_fit_temp_limit}'))

    fig.update_layout(xaxis_title='Temp /mK', yaxis_title='Theta /mV',
                      title=f'Dats{min(datnums)}-{max(datnums)}: Transition vs Fridge temp')

    return fig