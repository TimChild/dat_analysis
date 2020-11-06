from datetime import datetime
import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.DatObject.Attributes import Transition as T, Data as D, Entropy as E, Other as O, \
        Logs as L, AWG as A, SquareEntropy as SE

logger = logging.getLogger(__name__)


BASE_ATTRS = ['datnum', 'datname', 'dat_id', 'dattypes', 'date_initialized']

class DatHDF(object):
    """Overall Dat object which contains general information about dat, more detailed info should be put
    into DatAttribute classes. Everything in this overall class should be useful for 99% of dats
    """
    version = '1.0'
    """
    Version history
        1.0 -- HDF based save files
    """

    def __init__(self, datnum: int, datname, dat_hdf, Data=None, Logs=None, Entropy=None,
                 Transition=None, AWG=None, Other=None, SquareEntropy=None):
        """Constructor for dat"""
        self.version = DatHDF.version
        self.datnum = datnum
        self.datname = datname
        self.hdf = dat_hdf

        self.date_initialized = datetime.now().date()

        self.Data: D.NewData = Data
        self.Logs: L.NewLogs = Logs
        self.Entropy: E.NewEntropy = Entropy
        self.Transition: T.NewTransitions = Transition
        self.AWG: A.AWG = AWG
        self.Other: O.Other = Other
        self.SquareEntropy: SE.SquareEntropy = SquareEntropy

    def __del__(self):
        self.hdf.close()  # Close HDF when object is destroyed

