"""Utility functions for Dat"""


def get_id_from_val(data1d, value):
    """Returns closest ID of data to value, and returns true value"""
    return min(enumerate(data1d), key=lambda x: abs(x[1] - value))  # Gets the position of the