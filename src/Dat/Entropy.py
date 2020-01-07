import numpy as np

from src.CoreUtil import average_repeats


class Entropy:
    """
    Optional Dat attribute
        Represents components of the dat which are reserved to measurements of entropy
    """

    def __init__(self, dat, entx, enty=None):
        self.dat = dat
        self.entx = entx
        self.enty = enty

        self.entr = None
        self.entrav = None
        self.entangle = None
        self.calc_r(useangle=True)

    def calc_r(self, useangle=True):
        # calculate r data using either constant phase determined at largest value or larger signal
        # create averages - Probably this is still the best way to determine phase angle/which is bigger even if it's not repeat data
        xarray, entxav = average_repeats(self,
                                         returndata="entx")  # FIXME: Currently this requires Charge transition fits to be done first inside average_repeats
        xyarray, entyav = average_repeats(self, returndata="enty")
        sqr_x = np.square(entxav)
        sqr_y = np.square(entyav)
        sqr_xi = np.square(self.entx)  # non averaged data
        sqr_yi = np.square(self.enty)
        if max(np.abs(entxav)) > max(np.abs(entyav)):
            # if x is larger, take sign of x. Individual lines still take sign of averaged data otherwise it goes nuts
            sign = np.sign(entxav)
            signi = np.sign(self.entx)
            if np.abs(np.nanmax(entxav)) > np.abs(np.nanmin(entxav)):
                xmax = np.nanmax(entxav)
                xmaxi = np.nanargmax(entxav)
            else:
                xmax = np.nanmin(entxav)
                xmaxi = np.nanargmin(entxav)
            angle = np.arctan((entyav[xmaxi]) / xmax)
        else:
            # take sign of y data
            sign = np.sign(entyav)
            signi = np.sign(self.enty)
            if np.abs(np.nanmax(entyav)) > np.abs(np.nanmin(entyav)):
                ymax = np.nanmax(entyav)
                ymaxi = np.nanargmax(entyav)
            else:
                ymax = np.nanmin(entyav)
                ymaxi = np.nanargmin(entyav)
            angle = np.arctan(ymax / entxav[ymaxi])
        if useangle is False:
            self.entrav = np.multiply(np.sqrt(np.add(sqr_x, sqr_y)), sign)
            self.entr = np.multiply(np.sqrt(np.add(sqr_xi, sqr_yi)), signi)
            self.entangle = None
        elif useangle is True:
            self.entrav = np.array([x * np.cos(angle) + y * np.sin(angle) for x, y in zip(entxav, entyav)])
            self.entr = np.array([x * np.cos(angle) + y * np.sin(angle) for x, y in zip(self.entx, self.enty)])
            self.entangle = angle