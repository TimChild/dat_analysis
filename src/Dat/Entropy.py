import numpy as np
from typing import Tuple
from src.CoreUtil import average_repeats
import src.CoreUtil as CU
from src.CoreUtil import verbose_message
import src.config as cfg


class Entropy:
    """
    Optional Dat attribute
        Represents components of the dat which are reserved to measurements of entropy
    """
    __version = '1.0'
    def __init__(self, entx, mid_ids, enty=None):
        self.entx = np.array(entx)
        self.enty = np.array(enty)
        self.version = Entropy.__version

        self.entr = None
        self.entrav = None
        self.entangle = None
        self.calc_r(useangle=True)

    def calc_r(self, mid_ids=None, useangle=True):
        # calculate r data using either constant phase determined at largest value or larger signal
        # create averages - Probably this is still the best way to determine phase angle/which is bigger even if it's not repeat data

        if mid_ids is None:
            # region Verbose Entropy calc_r
            if cfg.verbose is True:
                verbose_message(f'Verbose[Entropy][calc_r] - No mid data provided for alignment')
            # endregion
            mid_ids = np.zeros(self.entx.shape[0])
        else:
            mid_ids = mid_ids

        entxav, entxav_err = CU.average_data(self.entx, mid_ids)
        entyav, entyav_err = CU.average_data(self.enty, mid_ids)
        sqr_x = np.square(entxav)
        sqr_y = np.square(entyav)
        sqr_x_orig = np.square(self.entx)
        sqr_y_orig = np.square(self.enty)

        x_max, y_max, which = _get_max_and_sign_of_max(entxav, entyav)  # Gets max of x and y at same location
        # and which was bigger
        angle = np.arctan(y_max / x_max)
        if which == 'x':
            sign = np.sign(entxav)
            sign_orig = np.sign(self.entx)
        elif which == 'y':
            sign = np.sign(entyav)
            sign_orig = np.sign(self.enty)
        else:
            raise ValueError('should have received "x" or "y"')

        if useangle is False:
            self.entrav = np.multiply(np.sqrt(np.add(sqr_x, sqr_y)), sign)
            self.entr = np.multiply(np.sqrt(np.add(sqr_x_orig, sqr_y_orig)), sign_orig)
            self.entangle = None
        elif useangle is True:
            self.entrav = np.array([x * np.cos(angle) + y * np.sin(angle) for x, y in zip(entxav, entyav)])
            self.entr = np.array([x * np.cos(angle) + y * np.sin(angle) for x, y in zip(self.entx, self.enty)])
            self.entangle = angle


def _get_max_and_sign_of_max(x, y) -> Tuple[float, float, np.array]:
    """Returns value of x, y at the max position of the larger of the two and which was larger...
     i.e. x and y value at index=10 if max([x,y]) is x at x[10] and 'x' because x was larger"""

    if max(np.abs(x)) > max(np.abs(y)):
        which = 'x'
        x_max, y_max = _get_values_at_max(x, y)
    else:
        which = 'y'
        y_max, x_max = _get_values_at_max(y, x)
    return x_max, y_max, which


def _get_values_at_max(larger, smaller) -> Tuple[float, float]:
    """Returns values of larger and smaller at position of max in larger"""
    if np.abs(np.nanmax(larger)) > np.abs(np.nanmin(larger)):
        large_max = np.nanmax(larger)
        index = np.nanargmax(larger)
    else:
        large_max = np.nanmin(larger)
        index = np.nanargmin(larger)
    small_max = smaller[index]
    return large_max, small_max
