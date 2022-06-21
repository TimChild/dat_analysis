"""
General functions for building a nicely standardized HDF file from experiment files.
Stuff in here should work quite generally for most experiment files. Really specific stuff (e.g. fixing how speicific dats have been saved should be done outside of this package)


Note:
It is not necessary to use anyhthing from this file to create the standardized HDF files as long as you stick to the correct conventions for the HDF file.
"""
from typing import Optional


def check_existing_hdf(path_to_exp_dat: str, path_to_hdfs: Optional[str] = None):
    """
    Using the path to the experiment dat file (something like <computer name>/<user name>/<exp_name>/<dat file>) check
    to see if a standardized HDF file has already been stored locally at path_to_hdfs, or the location in config file if
    not provided.

    Args:
        path_to_exp_dat: Full path to Dat file (something like <computer name>/<user name>/<exp_name>/<dat file>)
        path_to_hdfs: Path to where local HDFs should be stored

    Returns:

    """
    pass

