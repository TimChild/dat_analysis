import src.Configs.Main_Config as cfg
from src import CoreUtil as CU
import os
import h5py
import numpy as np
import lmfit as lm
import logging

logger = logging.getLogger(__name__)


def get_dat_hdf_path(dat_id, hdfdir_path, overwrite=False):
    file_path = os.path.join(hdfdir_path, dat_id + '.h5')
    if os.path.exists(file_path):
        if overwrite is True:
            os.remove(file_path)
        else:
            raise FileExistsError(f'HDF file already exists for {dat_id} at {hdfdir_path}. Use "overwrite=True" to overwrite')
    if not os.path.exists(file_path):  # make empty file then return path
        hdfdir_path, _ = os.path.split(file_path)
        os.makedirs(hdfdir_path, exist_ok=True)  # Ensure directory exists
        f = h5py.File(file_path, 'w')  # Init a HDF file
        f.close()
    return file_path


PARAM_KEYS = ['name', 'value', 'vary', 'min', 'max', 'expr', 'brute_step']


def params_to_HDF(params: lm.Parameters, group: h5py.Group):
    group.attrs['description'] = "Single Parameters of fit"
    for key in params.keys():
        par = params[key]
        par_group = group.require_group(key)
        par_group.attrs['description'] = "Single Param"
        for par_key in PARAM_KEYS:
            attr_val = getattr(par, par_key, np.nan)
            attr_val = attr_val if attr_val is not None else np.nan
            par_group.attrs[par_key] = attr_val
        par_group.attrs['init_value'] = getattr(par, 'init_value', np.nan)
        par_group.attrs['stderr'] = getattr(par, 'stderr', np.nan)
    pass


def params_from_HDF(group) -> lm.Parameters:
    params = lm.Parameters()
    for key in group.keys():
        if isinstance(group[key], h5py.Group) and group[key].attrs.get('description', None) == 'Single Param':
            par_group = group[key]
            par_vals = [par_group.attrs.get(par_key, None) for par_key in PARAM_KEYS]
            par_vals = [v if not (isinstance(v, float) and np.isnan(v)) else None for v in par_vals]
            params.add(*par_vals)  # create par
            par = params[key]  # Get single par
            par.stderr = par_group.attrs.get('stderr', None)
            par.value = par.init_value  # Because the saved value was actually final value, but inits into init_val
            par.init_value = par_group.attrs.get('init_value', None)  # I save init_value separately
            for par_key in PARAM_KEYS+['stderr', 'init_value']:
                if getattr(par, par_key) == np.nan:  # How I store None in HDF
                    setattr(par, par_key, None)
    return params


