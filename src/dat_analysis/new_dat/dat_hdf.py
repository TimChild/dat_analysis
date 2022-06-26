"""
2022-06: Trying to make a simpler Dat interface.

The aim of the DatHDF class is to provide an easy interface to the HDF files that are made here (i.e. not the experiment
files directly which differ too much from experiment to experiment)
"""
import os.path

import importlib.machinery
import re
from typing import Callable, Optional
import logging

from .build_dat_hdf import check_hdf_meets_requirements, default_exp_to_hdf

logger = logging.getLogger(__name__)

from .data_attr import Data
from .logs_attr import Logs

from ..hdf_util import HDFFileHandler
from .new_dat_util import get_local_config


class DatHDF:
    def __init__(self, hdf_path: str, mode='r'):
        self._hdf_path = hdf_path
        self.mode = mode
        passed_checks, message = check_hdf_meets_requirements(hdf_path)
        if not passed_checks:
            raise Exception(f'DatHDF at {hdf_path} does not meet requirements:\n{message}')

        self.Logs = Logs(hdf_path, '/Logs')
        self.Data = Data(hdf_path, '/Data')
        # self.Analysis = ... maybe add some shortcuts to standard analysis stuff? Fitting etc

    @property
    def hdf_read(self):
        """Explicitly open hdf for reading

        Examples:
            with dat.hdf_read as f:
                data = f['data'][:]
        """
        return HDFFileHandler(self._hdf_path, 'r')  # with self.hdf_read as f: ...

    @property
    def hdf_write(self):
        """
        Explicitly open hdf for writing

        Examples:
            with dat.hdf_write as f:
                f['data'] = np.array([1, 2, 3])
        """
        return HDFFileHandler(self._hdf_path, 'r+')  # with self.hdf_write as f: ...

    def __enter__(self):
        """
        Use dat with whatever dat.mode for read and/or writing

        Examples:
            with dat as f:
                data = f['data'][:]
                f.attrs['new_attr'] = 'new'  # Only if dat.mode is 'r+' or similar
        """
        self._handler = HDFFileHandler(self._hdf_path, self.mode)
        return self._handler.new()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._handler.previous()


def get_dat(datnum: Optional[int] = None,
            host_name = None, user_name = None, experiment_name = None,
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

    host_name = host_name if host_name else config['loading']['default_host_name']
    user_name = user_name if user_name else config['loading']['default_user_name']
    experiment_name = experiment_name if experiment_name else config['loading']['default_experiment_name']

    # Get path to directory containing datXX.h5 files
    exp_path = os.path.join(host_name, user_name, experiment_name)

    # Get path to specific datXX.h5 file and check it exists
    measurement_data_path = config['loading']['path_to_measurement_data']
    if raw is True:
        filepath = os.path.join(measurement_data_path, exp_path, f'dat{datnum}_RAW.h5')
    else:
        filepath = os.path.join(measurement_data_path, exp_path, f'dat{datnum}.h5')
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
    config = get_local_config()

    # Figure out path to DatHDF (existing or not)
    if override_save_path is None:
        match = re.search(r'measurement[-_]data[\\:/]+(.+)', experiment_data_path)
        after_measurement_data = match.groups()[0] if match else os.path.split(experiment_data_path)[1]
        save_path = os.path.join(config['loading']['path_to_save_directory'], after_measurement_data)
    elif isinstance(override_save_path, str):
        if os.path.isdir(override_save_path):
            raise IsADirectoryError(f'To override_save_path, must specify a full path to a file not a directory. Got ({override_save_path})')
        save_path = override_save_path
    else:
        raise ValueError(f"If providing 'override_save_path' it should be a path string. Got ({override_save_path}) instead")

    # If already existing, return or delete if overwriting
    if os.path.exists(save_path) and os.path.isfile(save_path):
        if overwrite:
            os.remove(save_path)
        else:
            return DatHDF(hdf_path=save_path, mode='r')

    # If not already returned, then create new standard DatHDF file from non-standard datXX.h5 file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if override_exp_to_hdf is not None:  # Use the specified function to convert
        override_exp_to_hdf(experiment_data_path, save_path, **loading_kwargs)
    elif config['loading']['path_to_python_load_file']:  # Use the file specified in config to convert
        # module = importlib.import_module(config['loading']['path_to_python_load_file'])
        module = importlib.machinery.SourceFileLoader('python_load_file', config['loading']['path_to_python_load_file']).load_module()
        fn = module.create_standard_hdf
        fn(experiment_data_path, save_path, **loading_kwargs)
    else:  # Do a basic default convert
        default_exp_to_hdf(experiment_data_path, save_path)

    # Return a DatHDF object from the standard DatHDF.h5 file
    return DatHDF(hdf_path=save_path, mode='r')





