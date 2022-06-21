"""
General utility functions related to the new simpler dat HDF interface
"""


def get_local_config() -> dict:
    """Get the configuration file saved on the machine (i.e. default place to store HDF files, default place to find
    experiment files, etc)"""

    # Look for a file stored in a location that should always be accessible (probably temp/...)
    return {}