from typing import NamedTuple
import logging

logger = logging.getLogger(__name__)


def data_to_NamedTuple(data: dict, named_tuple) -> NamedTuple:
    """Given dict of key: data and a named_tuple with the same keys, it returns the filled NamedTuple
    If data is not stored then a cfg._warning string is set"""
    tuple_dict = named_tuple.__annotations__  # Get ordered dict of keys of namedtuple
    for key in tuple_dict.keys():  # Set all values to None so they will default to that if not entered
        tuple_dict[key] = None
    for key in set(data.keys()) & set(tuple_dict.keys()):  # Enter valid keys values
        tuple_dict[key] = data[key]
    if set(data.keys()) - set(tuple_dict.keys()):  # If there is something left behind
        logger.warning(f'data keys not stored: {set(data.keys()) - set(tuple_dict.keys())}')
    ntuple = named_tuple(**tuple_dict)
    return ntuple


