from typing import Optional, Tuple

import numpy as np

from src.CoreUtil import data_row_name_append
from src import UsefulFunctions as U
from src.AnalysisTools.general_fitting import FitInfo, _get_transition_fit_func_params, calculate_transition_only_fit
from src.DatObject.Make_Dat import get_dat


def do_transition_only_calc(datnum, save_name: str,
                            theta=None, gamma=None, width=None, t_func_name='i_sense_digamma',
                            center_func: Optional[str] = None,
                            csq_mapped=False, data_rows: Tuple[Optional[int], Optional[int]] = (None, None),
                            centering_threshold: float = 1000,
                            experiment_name: Optional[str] = None,
                            overwrite=False) -> FitInfo:
    """
    Do calculations on Transition only measurements. Can be run multiprocessed
    Args:
        datnum ():
        save_name (): Name to save fits under in dat.Transition...
        theta (): If set, theta is forced to this value
        gamma (): If set, gamma is forced to this value
        width (): Width of fitting around center of transition
        t_func_name (): Name of fitting function to use for final averaged fit
        center_func (): Name of fitting function to use for centering data
        csq_mapped (): Whether to use CSQ mapped data
        data_rows (): Optionally select only certain rows to fit
        centering_threshold (): If dat.Logs.dacs['ESC'] is below this value, centering will happen
        experiment_name (): which cooldown basically e.g. FebMar21
        overwrite (): Whether to overwrite existing fits

    Returns:
        FitInfo: Returns fit, but fit is also saved in dat.Transition
    """
    dat = get_dat(datnum, exp2hdf=experiment_name)
    print(f'Working on {datnum}')

    if csq_mapped:
        name = 'csq_mapped'
        data_group_name = 'Data'
    else:
        name = 'i_sense'
        data_group_name = 'Transition'

    data = dat.Data.get_data(f'{name}_avg{data_row_name_append(data_rows)}', data_group_name=data_group_name,
                             default=None)
    if data is None or overwrite:

        s, f = data_rows
        rows = range(s if s else 0, f if f else dat.Data.get_data('y').shape[0])  # For saving correct row fit

        x = dat.Data.get_data('x', data_group_name=data_group_name)
        data = dat.Data.get_data(name, data_group_name=data_group_name)[s:f]

        # For centering if data does not already exist or overwrite is True
        if dat.Logs.dacs['ESC'] < centering_threshold:
            func_name = center_func if center_func is not None else t_func_name
            func, params = _get_transition_fit_func_params(x=x, data=np.mean(data, axis=0),
                                                           t_func_name=func_name,
                                                           theta=theta, gamma=gamma)

            center_fits = [dat.Transition.get_fit(which='row', row=row, name=f'{name}:{func_name}',
                                                  fit_func=func, initial_params=params,
                                                  data=d, x=x,
                                                  check_exists=False,
                                                  overwrite=overwrite) for row, d in zip(rows, data)]

            centers = [fit.best_values.mid for fit in center_fits]
        else:
            centers = [0] * len(dat.Data.get_data('y'))
        data_avg, x_avg = U.mean_data(x=x, data=data, centers=centers, method='linear', return_x=True)
        for d, n in zip([data_avg, x_avg], [name, 'x']):
            dat.Data.set_data(data=d, name=f'{n}_avg{data_row_name_append(data_rows)}',
                              data_group_name=data_group_name)

    x = dat.Data.get_data(f'x_avg{data_row_name_append(data_rows)}', data_group_name=data_group_name)
    data = dat.Data.get_data(f'{name}_avg{data_row_name_append(data_rows)}', data_group_name=data_group_name)

    try:
        fit = calculate_transition_only_fit(datnum, save_name=save_name, t_func_name=t_func_name, theta=theta,
                                            gamma=gamma, x=x, data=data, width=width,
                                            experiment_name=experiment_name,
                                            overwrite=overwrite)
    except (TypeError, ValueError):
        print(f'Dat{dat.datnum}: Fit Failed. Returning None')
        fit = None
    return fit