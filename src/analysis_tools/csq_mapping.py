from typing import Optional, Tuple, List, Callable
import logging
import numpy as np
from progressbar import progressbar
from scipy.interpolate import interp1d

from src.core_util import data_row_name_append, get_data_index
from src import useful_functions as U
from src.dat_object.make_dat import get_dats

logger = logging.getLogger(__name__)


def setup_csq_dat(csq_datnum: int, experiment_name: Optional[str] = None, overwrite=False):
    """Run this on the CSQ dat once to set up the interpolating datasets"""
    from src.dat_object.make_dat import get_dat
    csq_dat = get_dat(csq_datnum, exp2hdf=experiment_name)
    if any([name not in csq_dat.Data.keys for name in ['csq_x', 'csq_data']]) or overwrite:
        csq_data = csq_dat.Data.get_data('cscurrent')
        csq_x = csq_dat.Data.get_data('x')

        in_range = np.where(np.logical_and(csq_data < 16, csq_data > 1))
        cdata = U.decimate(csq_data[in_range], measure_freq=csq_dat.Logs.measure_freq, numpnts=100)
        cx = U.get_matching_x(csq_x[in_range], cdata)  # Note: cx is the target y axis for cscurrent

        # Remove a few nans from beginning and end (from decimating)
        cx = cx[np.where(~np.isnan(cdata))]
        cdata = cdata[np.where(~np.isnan(cdata))]

        cx = cx - cx[U.get_data_index(cdata, 7.25,
                                      is_sorted=True)]  # Make 7.25nA be the zero point since that's where I try center the CS

        csq_dat.Data.set_data(cx, name='csq_x')
        csq_dat.Data.set_data(cdata, name='csq_data')


def get_csq_mapper(csq_datnum: int) -> interp1d:
    from src.dat_object.make_dat import get_dat
    csq_dat = get_dat(csq_datnum)
    if any([name not in csq_dat.Data.keys for name in ['csq_x', 'csq_data']]):
        raise RuntimeError(f'CSQ_Dat{csq_datnum}: Has not been initialized, run setup_csq_dat({csq_datnum}) first')
    cx = csq_dat.Data.get_data('csq_x')
    cdata = csq_dat.Data.get_data('csq_data')
    interper = interp1d(cdata, cx, kind='linear', bounds_error=False, fill_value=np.nan)
    return interper


def calculate_csq_map(datnum: int, experiment_name: Optional[str] = None, csq_datnum: Optional[int] = None,
                      overwrite=False):
    """Do calculations to generate data in csq gate from i_sense using csq trace from csq_dat"""
    from src.dat_object.make_dat import get_dat
    if csq_datnum is None:
        csq_datnum = 1619
    dat = get_dat(datnum, exp2hdf=experiment_name)
    if 'csq_mapped' not in dat.Data.keys or overwrite:
        odata = dat.Data.get_data('i_sense')
        ndata = csq_map_data(odata, csq_datnum)
        dat.Data.set_data(ndata, name='csq_mapped')
    return dat.Data.get_data('csq_mapped')


def csq_map_data(data: np.ndarray, csq_datnum: int) -> np.ndarray:
    """Maps data using linear interpolation of a csq_x and data trace"""
    interper = get_csq_mapper(csq_datnum)
    ndata = interper(data)
    return ndata


def _calculate_csq_avg(datnum: int, centers=None,
                       data_rows: Optional[Tuple[Optional[int], Optional[int]]] = None,
                       experiment_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes existing csq_mapped i_sense data and averages it using centers
    Args:
        datnum ():
        centers ():
        data_rows ():
        experiment_name ():

    Returns:

    """
    from src.dat_object.make_dat import get_dat
    dat = get_dat(datnum, exp2hdf=experiment_name)
    if centers is None:
        logger.warning(f'Dat{dat.datnum}: No centers passed for averaging CSQ mapped data')
        raise ValueError('Need centers')

    x = dat.Data.get_data('x')
    data = dat.Data.get_data('csq_mapped')[data_rows[0]: data_rows[1]]

    avg_data, csq_x_avg = U.mean_data(x, data, centers, method='linear', return_x=True)

    dat.Data.set_data(avg_data, f'csq_mapped_avg{data_row_name_append(data_rows)}')
    dat.Data.set_data(csq_x_avg, f'csq_x_avg{data_row_name_append(data_rows)}')
    return avg_data, csq_x_avg


def calculate_csq_mapped_avg(datnum: int, csq_datnum: Optional[int] = None,
                             centers: Optional[List[float]] = None,
                             data_rows: Tuple[Optional[int], Optional[int]] = (None, None),
                             experiment_name: Optional[str] = None,
                             overwrite=False):
    """Calculates CSQ mapped data, and averaaged data and saves in dat.Data....
    Note: Not really necessary to have avg data calculated for square entropy, because running SE will average and
    center data anyway
    """
    from src.dat_object.make_dat import get_dat
    dat = get_dat(datnum, exp2hdf=experiment_name)
    if 'csq_mapped' not in dat.Data.keys or overwrite:
        calculate_csq_map(datnum, csq_datnum=csq_datnum, overwrite=overwrite)

    if f'csq_mapped_avg{data_row_name_append(data_rows)}' not in dat.Data.keys or overwrite:
        _calculate_csq_avg(datnum, centers=centers, data_rows=data_rows)

    return dat.Data.get_data(f'csq_mapped_avg{data_row_name_append(data_rows)}'), \
           dat.Data.get_data(f'csq_x_avg{data_row_name_append(data_rows)}')


def multiple_csq_maps(csq_datnums: List[int], datnums_to_map: List[int],
                      sort_func: Optional[Callable] = None,
                      warning_tolerance: Optional[float] = None,
                        experiment_name: Optional[str] = None,
                      overwrite=False) -> True:
    """
    Using `csq_datnums`, will map all `datnums_to_map` based on whichever csq dat has is closest based on `sort_func`
    Args:
        csq_datnums (): All the csq datnums which might be used to do csq mapping (only closest based on sort_func will be used)
        datnums_to_map (): All the data datnums which should be csq mapped
        sort_func (): A function which takes dat as the argument and returns a float or int
        warning_tolerance: The max distance a dat can be from the csq dats based on sort_func without giving a warning
        experiment_name (): which cooldown basically e.g. FebMar21
        overwrite (): Whether to overwrite prexisting mapping stuff

    Returns:
        bool: Success
    """
    if sort_func is None:
        sort_func = lambda dat: dat.Logs.fds['ESC']
    csq_dats = get_dats(csq_datnums, exp2hdf=experiment_name)
    csq_dict = {sort_func(dat): dat for dat in csq_dats}
    transition_dats = get_dats(datnums_to_map, exp2hdf=experiment_name)

    for num in progressbar(csq_datnums):
        setup_csq_dat(num, overwrite=overwrite)

    csq_sort_vals = list(csq_dict.keys())
    for dat in progressbar(transition_dats):
        closest_val = csq_sort_vals[get_data_index(np.array(csq_sort_vals), sort_func(dat))]
        if warning_tolerance is not None:
            if (dist := abs(closest_val - sort_func(dat))) > warning_tolerance:
                logging.warning(f'Dat{dat.datnum}: Closest CSQ dat has distance {dist:.2f} from Dat based on sort_func')
        calculate_csq_map(dat.datnum, experiment_name=experiment_name, csq_datnum=csq_dict[closest_val].datnum,
                          overwrite=overwrite,
                          )
    return True