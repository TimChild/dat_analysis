from typing import Optional, Tuple, List
import logging

import numpy as np
from scipy.interpolate import interp1d

from src.CoreUtil import data_row_name_append
from src import UsefulFunctions as U
from src.DatObject.Make_Dat import get_dat

logger = logging.getLogger(__name__)


def setup_csq_dat(csq_datnum: int, experiment_name: Optional[str] = None, overwrite=False):
    """Run this on the CSQ dat once to set up the interpolating datasets"""
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


def calculate_csq_map(datnum: int, experiment_name: Optional[str] = None, csq_datnum: Optional[int] = None,
                      overwrite=False):
    """Do calculations to generate data in csq gate from i_sense using csq trace from csq_dat"""
    if csq_datnum is None:
        csq_datnum = 1619
    dat = get_dat(datnum, exp2hdf=experiment_name)
    csq_dat = get_dat(csq_datnum, exp2hdf=experiment_name)

    if 'csq_mapped' not in dat.Data.keys or overwrite:
        if any([name not in csq_dat.Data.keys for name in ['csq_x', 'csq_data']]):
            raise RuntimeError(f'CSQ_Dat{csq_datnum}: Has not been initialized, run setup_csq_dat({csq_datnum}) first')
        cx = csq_dat.Data.get_data('csq_x')
        cdata = csq_dat.Data.get_data('csq_data')

        interper = interp1d(cdata, cx, kind='linear', bounds_error=False, fill_value=np.nan)
        odata = dat.Data.get_data('i_sense')

        ndata = interper(odata)

        dat.Data.set_data(ndata, name='csq_mapped')
    return dat.Data.get_data('csq_mapped')


def _calculate_csq_avg(datnum: int, centers=None,
                       data_rows: Optional[Tuple[Optional[int], Optional[int]]] = None,
                       experiment_name: Optional[str] = None) -> Tuple[
    np.ndarray, np.ndarray]:
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
    dat = get_dat(datnum, exp2hdf=experiment_name)
    if 'csq_mapped' not in dat.Data.keys or overwrite:
        calculate_csq_map(datnum, csq_datnum=csq_datnum, overwrite=overwrite)

    if f'csq_mapped_avg{data_row_name_append(data_rows)}' not in dat.Data.keys or overwrite:
        _calculate_csq_avg(datnum, centers=centers, data_rows=data_rows)

    return dat.Data.get_data(f'csq_mapped_avg{data_row_name_append(data_rows)}'), \
           dat.Data.get_data(f'csq_x_avg{data_row_name_append(data_rows)}')