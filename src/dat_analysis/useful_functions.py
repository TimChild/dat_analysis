"""
Simple general functions that are useful in analysis but NOT used in any other modules

WARNING: IMPORTING FROM HERE IN ANY OTHER MODULE IS LIKELY TO CAUSE CIRCULAR IMPORT ISSUES

NOTE: If any of the functions here become useful in any other module in dat_analysis, the function should be moved to
core_util.py and then imported here for backwards compatability and ease of access"""

import logging
from deprecation import deprecated
import base64
from IPython.display import Image
import scipy.signal

logger = logging.getLogger(__name__)

# Make these easy to access from one location
from .core_util import (
    get_data_index,
    get_matching_x,
    edit_params,
    sig_fig,
    decimate,
    FIR_filter,
    get_sweeprate,
    bin_data,
    get_bin_size,
    mean_data,
    center_data,
    resample_data,
    ensure_list,
    order_list,
    my_round,
)

from .hdf_util import NotFoundInHdfError


def mm(graph):
    """Make a mermaid graph (in Jupyter notebooks)
    Examples:
        mm('''
            graph LR;
                A--> B & C & D;
                B--> A & E;
                C--> A & E;
                D--> A & E;
                E--> B & C & D;
            ''')
    """
    graphbytes = graph.encode("ascii")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    image = Image(url="https://mermaid.ink/img/" + base64_string)
    return image


def set_default_logging(level_override=None):
    # logging.basicConfig(level=logging.INFO, format=f'%(threadName)s %(funcName)s %(lineno)d %(message)s')
    # logging.basicConfig(level=logging.INFO, force=True, format=f'%(levelname)s:%(module)s:%(lineno)d:%(funcName)s:%(message)s')
    root_logger = logging.getLogger()
    root_logger.handlers = []  # Probably a bad thing to be doing...
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        f"%(thread)d:%(process)d:%(levelname)s:%(module)s:%(lineno)d:%(funcName)s:%(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    if level_override:
        root_logger.setLevel(level_override)


@deprecated(deprecated_in="3.2.0", details="Part of the Data class now")
def power_spectrum(data, meas_freq, normalization=1):
    """
    Computes power spectrum and returns (freq, power spec)
    Args:
        data (): 1D data to calculate spectrum for
        meas_freq (): Frequency of measurement (not sample rate)
        normalization (): Multiplies data before calculating

    Returns:

    """
    freq, power = scipy.signal.periodogram(data * normalization, fs=meas_freq)
    return freq, power


def dac_step_freq(x_array=None, freq=None, dat=None):
    """Calculates the frequency of the true DAC steps (i.e. 20V/2^16 is smallest DAC step size)"""
    if dat:
        assert all([x_array is None, freq is None])
        x_array = dat.Data.x_array
        freq = dat.Logs.Fastdac.measure_freq

    full_x = abs(x_array[-1] - x_array[0])
    num_x = len(x_array)
    min_step = 20000 / 2**16
    req_step = full_x / num_x
    step_every = min_step / req_step
    step_t = step_every / freq
    step_hz = 1 / step_t
    return step_hz


