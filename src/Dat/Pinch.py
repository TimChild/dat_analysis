import numpy as np
from typing import Tuple, List
from src.CoreUtil import average_repeats
import src.CoreUtil as CU
from src.CoreUtil import verbose_message
import src.config as cfg
import lmfit as lm
import pandas as pd
import matplotlib.pyplot as plt
import src.Dat.DatAttribute as DA
import src.Dat.PlottingFunctions as PF

# TODO: Finish this... I had to give up part way due to lack of time. 27/01/2020


class Pinch(DA.DatAttribute):
    """
    Optional Dat attribute
        For pinch off measurements
    """
    __version = '1.0'

    def __init__(self, x_array, conductance: np.array, current: np.array = None, bias: float = None):
        assert conductance.ndim == 1  # assert 1D data
        if current is not None:
            assert current.ndim == 1
        self.x_array = x_array
        self.data = conductance
        self.version = Pinch.__version
        self.pinch = self.pinch_value()

    def pinch_value(self):
        """Returns highest x value for which current < 0.1nA"""
        pinched_ids = [self.data.index(val) for val in self.data if val < 0.1]
        pinch_value = np.nanmax([self.x_array[i] for i in pinched_ids])
        return pinch_value

    def recalculate(self):
        self.version = Pinch.__version
        self.pinch = self.pinch_value()
        
    def plot(self, ax:plt.Axes=None):
        if ax is None:
            ax = plt.subplots(1,1)
        PF.display_1d(self.x_array, self.data, ax=ax, xlabel="Gate /mV", ylabel="Conductance (e^2/h")
        return ax
