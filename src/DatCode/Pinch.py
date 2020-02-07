import numpy as np
import matplotlib.pyplot as plt
import src.DatCode.DatAttribute as DA
import src.PlottingFunctions as PF
import src.CoreUtil as CU
import pandas as pd


# TODO: Finish this... I had to give up part way due to lack of time. 27/01/2020


class Pinch(DA.DatAttribute):
    """
    Optional Dat attribute
        For pinch off measurements
    """
    __version = '1.0'

    def __init__(self, x_array, current: np.array, conductance: np.array = None, bias: float = None):
        if current is not None:
            assert current.ndim == 1
        self._current = current
        self._x_array = x_array
        self.version = Pinch.__version
        if self._current is not None:
            self.pinch = self.pinch_value()

        self._plot_current_args = None

    def pinch_value(self):
        """Returns highest x value for which current < 0.1nA"""
        print("Not finished 'pinch_value'")
        pinched_ids = [i for i, val in enumerate(self._current) if val < 0.1]
        if len(pinched_ids) != 0:
            pinch_value = np.nanmax([self._x_array[i] for i in pinched_ids])
        else:
            print('Assuming pinch measurement stopped at cutoff higher than 0.1nA')
            pd_current = pd.Series(self._current)  # Make pandas array to get last non NaN from data which is probably
            # where the 'pinch' was
            pinch_value = self._x_array[pd_current.last_valid_index()]
        return pinch_value

    def recalculate(self):
        self.version = Pinch.__version
        self.pinch = self.pinch_value()

    def plot_current(self, ax: plt.Axes = None, default=False, **kwargs):
        if len(kwargs) == 0 and self._plot_current_args is not None and default is False:
            kwargs = self._plot_current_args

        CU.set_kwarg_if_none('y_label', 'Current /nA', kwargs)  # only sets if not passed in as argument
        CU.set_kwarg_if_none('x_label', 'Gate /mV', kwargs)
        self._plot_current_args = kwargs  # Saves last used kwargs to be default next time
        ax = PF.display_1d(self._x_array, self._current, ax=ax, **kwargs)
        return ax
