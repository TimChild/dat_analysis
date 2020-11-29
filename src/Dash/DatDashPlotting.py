"""
This is where all general dat plotting functions should live... To use in other pages, import the more general plotting
function from here, and make a little wrapper plotting function which calls with the relevant arguments
"""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF


class DatPlotter:
    def __init__(self, dat: DatHDF):
        self.dat = dat

    def axis_labels(self):
        return self.dat.Logs.xlabel, self.dat.Logs.ylabel

