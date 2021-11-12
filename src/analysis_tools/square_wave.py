from typing import Optional, Tuple, Union

import numpy as np

from src.dat_object.dat_hdf import DatHDF
from src.dat_object.make_dat import get_dat
from src.dat_object.attributes.SquareEntropy import square_wave_time_array, Output
import src.useful_functions as U


def get_setpoint_indexes_from_times(dat: DatHDF,
                                    start_time: Optional[float] = None,
                                    end_time: Optional[float] = None) -> Tuple[Union[int, None], Union[int, None]]:
    """
    Gets the indexes of setpoint_start/fin from a start and end time in seconds
    Args:
        dat (): Dat for which this is being applied
        start_time (): Time after setpoint change in seconds to start averaging setpoint
        end_time (): Time after setpoint change in seconds to finish averaging setpoint
    Returns:
        start, end indexes
    """
    setpoints = [start_time, end_time]
    setpoint_times = square_wave_time_array(dat.SquareEntropy.square_awg)
    sp_start, sp_fin = [U.get_data_index(setpoint_times, sp) for sp in setpoints]
    return sp_start, sp_fin


def set_transition_data_to_cold_only(datnum: int, setpoint_start_ms: float) -> int:
    """
    Set data in dat.Transition to be that of the cold part of a square wave heated scan only (ignoring
    setpoint_start_ms after each setpoint change)

    Note: overwrites data in dat.Transition

    Args:
        datnum ():
        setpoint_start_ms ():

    Returns:
        int: datnum, just so that it is easier to check which run successfully if multithreaded for example

    """
    dat = get_dat(datnum)
    sp_start_id, _ = get_setpoint_indexes_from_times(dat, setpoint_start_ms, None)
    pp = dat.SquareEntropy.get_ProcessParams(setpoint_start=sp_start_id)
    out = dat.SquareEntropy.get_row_only_output(name='default', process_params=pp, calculate_only=True)

    # This overwrites the data in dat.Transition
    dat.Transition.data = dat.SquareEntropy.get_transition_part(which='row', row=None,  # Get all rows
                                                                part='cold', data=out.cycled)

    dat.Transition.x = U.get_matching_x(dat.Transition.x, dat.Transition.data)

    return dat.datnum  # Just a simple return to make it easier to see which ran successfully


def data_from_output(o: Output, w: str):
    if w == 'i_sense_cold':
        return np.nanmean(o.averaged[(0, 2,), :], axis=0)
    elif w == 'i_sense_hot':
        return np.nanmean(o.averaged[(1, 3,), :], axis=0)
    elif w == 'entropy':
        return o.average_entropy_signal
    elif w == 'dndt':
        return o.average_entropy_signal
    elif w == 'integrated':
        d = np.nancumsum(o.average_entropy_signal)
        return d / np.nanmax(d)
    else:
        return None