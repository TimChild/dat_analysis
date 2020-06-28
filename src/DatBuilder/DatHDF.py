from datetime import datetime
import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.DatAttributes import Transition as T, Entropy as E, DCbias as DC, Logs as L, Instruments as I, Data as D

logger = logging.getLogger(__name__)


BASE_ATTRS = ['datnum', 'datname', 'dat_id', 'dattypes', 'date_initialized']


# noinspection PyPep8Naming
class DatHDF(object):
    """Overall Dat object which contains general information about dat, more detailed info should be put
    into DatAttribute classes. Everything in this overall class should be useful for 99% of dats
    """
    version = '1.0'
    """
    Version history
        1.0 -- HDF based save files
    """

    def __init__(self, datnum: int, datname, dat_hdf, Data=None, Logs=None, Instruments=None, Entropy=None,
                 Transition=None, DCbias=None, AWG=None):
        """Constructor for dat"""
        self.version = DatHDF.version
        self.config_name = 'No longer valid here'  # cfg.current_config.__name__.split('.')[-1]
        self.dattype = None
        self.datnum = datnum
        self.datname = datname
        self.hdf = dat_hdf

        self.date_initialized = datetime.now().date()

        self.Data: D.NewData = Data
        self.Logs: L.NewLogs = Logs
        self.Instruments: I.NewInstruments = Instruments
        self.Entropy: E.NewEntropy = Entropy
        self.Transition: T.NewTransitions = Transition
        self.DCbias: DC.NewDCBias = DCbias
        self.AWG: AWG.AWG = AWG

    def __del__(self):
        self.hdf.close()  # Close HDF when object is destroyed

############## FIGURE OUT WHAT TO DO WITH/WHERE TO PUT

# predicted frequencies in power spectrum from dac step size
# dx = np.mean(np.diff(x))
# dac_step = 20000/2**16  # 20000mV full range with 16bit dac
# step_freq = meas_freq/(dac_step/dx)
#
# step_freqs = np.arange(1, meas_freq/2/step_freq)*step_freq
#
# fig, ax = plt.subplots(1)
# PF.Plots.power_spectrum(deviation, 2538 / 2, 1, ax, label='Average_filtered')
#
# # step_freqs = np.arange(1, meas_freq / 2 / 60) * 60
#
# for f in step_freqs:
#     ax.axvline(f, color='orange', linestyle=':')