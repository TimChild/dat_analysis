"""This sets up what is accessible directly from dat_analysis.

This should be reserved for VERY commonly used functions ONLY

E.g. from dat_analysis import get_dat will work and the user doesn't need to know that it's tucked away in dat_analysis.dat.dat_hdf
"""
from .dat.dat_hdf import get_dat, get_dat_from_exp_filepath
from .analysis_tools.data import Data
