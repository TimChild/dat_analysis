"""
2022-06: Trying to make a simpler Dat interface.

The aim of the DatHDF class is to provide an easy interface to the HDF files that are made here (i.e. not the experiment
files directly which differ too much from experiment to experiment)
"""
import os.path
import tempfile

import importlib.machinery
import re
from typing import Callable, Optional
import logging

from .build_dat_hdf import check_hdf_meets_requirements, default_exp_to_hdf

logger = logging.getLogger(__name__)

from .data_attr import Data
from .logs_attr import Logs

from ..hdf_file_handler import HDF, GlobalLock
from .dat_util import get_local_config
from ..core_util import get_full_path


class DatHDF(HDF):
    def __init__(self, hdf_path: str):
        super().__init__(hdf_path)
        passed_checks, message = check_hdf_meets_requirements(hdf_path)
        if not passed_checks:
            raise Exception(f'DatHDF at {hdf_path} does not meet requirements:\n{message}')

        self.Logs = Logs(hdf_path, '/Logs')
        self.Data = Data(hdf_path, '/Data')
        # self.Analysis = ... maybe add some shortcuts to standard analysis stuff? Fitting etc

        self._datnum = None

    @property
    def datnum(self):
        if not self._datnum:
            with self.hdf_read as f:
                self._datnum = f.attrs.get('datnum', -1)
        return self._datnum


def get_dat(datnum: int,
            # host_name = None, user_name = None, experiment_name = None,
            host_name, user_name, experiment_name,
            raw=False, overwrite=False,
            override_save_path=None,
            **loading_kwargs):
    """
    Function to help with loading DatHDF object.

    Note: Not all arguments should be provided from (datnum, host_name, user_name, experiment_name)
        Either:
            - datnum only: Uses the 'current_experiment_path' in config.toml to decide where to look
            - datnum, host_name, user_name, experiment_name: Look for dat at 'path_to_measurement_data/host_name/user_name/experiment_name/dat{datnum}.h5'
        To load from direct data path, use 'get_dat_from_exp_filepath' instead.
        To load from already existing DatHDF.h5 without checking for existing experiment data etc, initialize DatHDF directly with DatHDF(hdf_path=filepath, mode='r')

    Args:
        datnum (): Optionally provide datnum
        host_name (): Optionally provide host_name of computer measurement was taken from (e.g. qdev-xld)
        user_name (): Optionally provide the user_name the data was stored under (i.e. matching how it appears in the server file structure)
        experiment_name (): Optionally provide the folder (or path to folder using '/' to join folders) of where to find data under host_name/user_name/... (e.g. 'tests/test1_date/')
        raw (): Bool to specify whether to load datXX_RAW.h5 or datXX.h5 (defaults to False)
        overwrite (): Whether to overwrite a possibly existing DatHDF (defaults to False)
        override_save_path (): Optionally override the path where the DatHDF will be looked for/stored
        **loading_kwargs (): Any other args that are needed in order to load the dat.h5

    Returns:
        DatHDF: A python object for easy interaction with a standardized HDF file.
    """
    config = get_local_config()
    # host_name = host_name if host_name else config['loading']['default_host_name']
    # user_name = user_name if user_name else config['loading']['default_user_name']
    # experiment_name = experiment_name if experiment_name else config['loading']['default_experiment_name']

    # Get path to directory containing datXX.h5 files
    exp_path = os.path.join(host_name, user_name, experiment_name)

    # Get path to specific datXX.h5 file and check it exists
    measurement_data_path = config['loading']['path_to_measurement_data']
    if raw is True:
        filepath = os.path.join(measurement_data_path, exp_path, f'dat{datnum}_RAW.h5')
    else:
        filepath = os.path.join(measurement_data_path, exp_path, f'dat{datnum}.h5')
    filepath = get_full_path(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'{filepath}')

    return get_dat_from_exp_filepath(filepath, overwrite=overwrite, override_save_path=override_save_path, **loading_kwargs)


def get_dat_from_exp_filepath(experiment_data_path: str, overwrite: bool=False, override_save_path: Optional[str]=None,
                              override_exp_to_hdf: Optional[Callable] = None, **loading_kwargs):
    """
    Get a DatHDF for given experiment data path... Uses experiment data path to decide where to save DatHDF if
    override_save_path not provided

    Args:
        experiment_data_path (): Path to the hdf file as saved at experiment time (i.e. likely some not quite standard format)
        overwrite (): Whether to overwrite an existing DatHDF file
        override_save_path (): Optionally provide a string path of where to save DatHDF (or look for existing DatHDF)
        override_exp_to_hdf (): Optionally provide a function which will be used to turn the experiment hdf file into the standardized DatHDF file.
            Function should take arguments (experiment_data_path, save_path, **kwargs) and need not return anything.
        **loading_kwargs (): Any other args that are needed in order to load the dat.h5

    Returns:

    """

    experiment_data_path = get_full_path(experiment_data_path)
    if override_save_path:
        override_save_path = get_full_path(override_save_path)

    # Figure out path to DatHDF (existing or not)
    if override_save_path is None:
        save_path = save_path_from_exp_path(experiment_data_path)
    elif isinstance(override_save_path, str):
        if os.path.isdir(override_save_path):
            raise IsADirectoryError(f'To override_save_path, must specify a full path to a file not a directory. Got ({override_save_path})')
        save_path = override_save_path
    else:
        raise ValueError(f"If providing 'override_save_path' it should be a path string. Got ({override_save_path}) instead")

    # If already existing, return or delete if overwriting
    if os.path.exists(save_path) and os.path.isfile(save_path) and not overwrite:
        return DatHDF(hdf_path=save_path)

    # If not already returned, then create new standard DatHDF file from non-standard datXX.h5 file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lock = GlobalLock(save_path+'.init.lock')  # Can't just use '.lock' as that is used by FileQueue
    # lock = GlobalLock(save_path+'.lock')
    with lock:  # Only one thread/process should be doing this for any specific save path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            if overwrite:
                os.remove(save_path)  # Possible thaht this was just created by another thread, but if trying to overwrite, still better to remove again
            else:
                return DatHDF(hdf_path=save_path)  # Must have been created whilst this thread was waiting
        if override_exp_to_hdf is not None:  # Use the specified function to convert
            override_exp_to_hdf(experiment_data_path, save_path, **loading_kwargs)
        elif config := get_local_config() and get_local_config()['loading']['path_to_python_load_file']:  # Use the file specified in config to convert
            # module = importlib.import_module(config['loading']['path_to_python_load_file'])
            module = importlib.machinery.SourceFileLoader('python_load_file', config['loading']['path_to_python_load_file']).load_module()
            fn = module.create_standard_hdf
            fn(experiment_data_path, save_path, **loading_kwargs)
        else:  # Do a basic default convert
            default_exp_to_hdf(experiment_data_path, save_path)

    # Return a DatHDF object from the standard DatHDF.h5 file
    return DatHDF(hdf_path=save_path)


def save_path_from_exp_path(experiment_data_path: str) -> str:
    config = get_local_config()
    match = re.search(r'measurement[-_]data[\\:/]*(.+)', experiment_data_path)
    after_measurement_data = match.groups()[0] if match else \
        re.match(r'[\\:/]*(.+)', os.path.splitdrive(experiment_data_path)[1]).groups()[
            0]  # TODO: better way to handle this? This could make some crazy file locations...
    save_path = get_full_path(os.path.join(config['loading']['path_to_save_directory'], os.path.normpath(after_measurement_data)))
    return save_path



