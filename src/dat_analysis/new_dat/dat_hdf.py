"""
2022-06: Trying to make a simpler Dat interface.

The aim of the DatHDF class is to provide an easy interface to the HDF files that are made here (i.e. not the experiment
files directly which differ too much from experiment to experiment)
"""
import os.path

import h5py
import json
import logging
logger = logging.getLogger(__name__)

from .data_attr import Data
from .logs_attr import Logs

from ..hdf_util import HDFFileHandler, NotFoundInHdfError
from .new_dat_util import get_local_config
from ..dat_object.attributes.logs import InitLogs


class DatHDF:
    def __init__(self, hdf_path: str, mode='r'):
        self._hdf_path = hdf_path
        self.mode = mode
        self._check_existing_hdf()  # Ensure it looks like the standard HDF dat file

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

    def _check_existing_hdf(self):
        """Check the hdf_path points to an HDF file that contains expected groups/attrs"""
        with self.hdf_read as f:
            keys = f.keys()
            for k in ['Logs', 'Data']:
                if k not in keys:
                    raise NotFoundInHdfError(f'Did not find {k} in {self._hdf_path}')


def get_dat(datnum: int, host_name = None, user_name = None, experiment_name = None, raw=False, overwrite=False, override_save_path=None):
    if None in (host_name, user_name, experiment_name):
        config = get_local_config()
        exp_path = config['current_experiment_path']
    elif None not in (host_name, user_name, experiment_name):
        exp_path = os.path.join(host_name, user_name, experiment_name)
        if not os.path.isdir(exp_path):
            raise NotADirectoryError(exp_path)
    else:
        raise ValueError(f'Must either provide all or none of [host_name, user_name, experiment_name]')

    if raw is True:
        filepath = os.path.join(exp_path, f'Dat{datnum}_RAW.h5')
    else:
        filepath = os.path.join(exp_path, f'Dat{datnum}.h5')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f'{filepath}')

    return get_dat_from_exp_filepath(filepath, overwrite=overwrite, override_save_path=override_save_path)


def get_dat_from_exp_filepath(experiment_data_path: str, overwrite=False, override_save_path=None):
    """Get a DatHDF for given experiment data path... Uses experiment data path to decide where to save DatHDF"""
    # Figure out path to DatHDF (existing or not)
    if override_save_path is None:
        config = get_local_config()
        save_path = os.path.join(config['path_to_save_directory'], os.path.dirname(os.path.splitdrive(experiment_data_path)[1]))
    else:
        save_path = override_save_path

    # If already existing, return or delete
    if os.path.exists(save_path):
        if overwrite:
            os.remove(save_path)
        else:
            return DatHDF(hdf_path=save_path, mode='r')

    # If not already returned, then create new DatHDF file and return
    default_exp_to_hdf(experiment_data_path, save_path)
    return DatHDF(hdf_path=save_path, mode='r')


def default_exp_to_hdf(exp_data_path: str, new_save_path: str):
    """Copy the relevant information from experiment Dat into standardized DatHDF

    Note: May be necessary to copy more stuff after depending on the state of the experiment hdf file (i.e. bad jsons
    or missing metadata)
    """
    if os.path.exists(new_save_path):
        raise FileExistsError(f'File already exists at {new_save_path}')

    with HDFFileHandler(exp_data_path, 'r') as o:
        with HDFFileHandler(new_save_path, 'w') as n:
            data_group = n.require_group('Data')
            logs_group = n.require_group('Logs')
            for k in o.keys():
                if isinstance(o[k], h5py.Dataset):
                    data_group[k] = o[k]

            if 'metadata' in o.keys() and 'sweep_logs' in o['metadata'].attrs.keys():
                sweeplogs_str = o['metadata'].attrs['sweep_logs']
                logs_group.attrs['sweep_logs_string'] = sweeplogs_str
                default_sort_sweeplogs(logs_group, sweeplogs_str)

            # TODO: Also ScanVars and exp_config?


def default_sort_sweeplogs(logs_group: h5py.Group, sweep_logs_str: str):
    """Convert the sweep_logs string into standard attributes in logs_group"""

    try:
        logs = json.loads(sweep_logs_str)
    except json.JSONDecodeError as e:
        logger.error(f'Error, skipping the making nice Logs: {e.msg}')
        logs = None

    if logs:
        try:
            # FastDAC
            if 'FastDAC' in logs.keys() or 'FastDAC 1' in logs.keys():
                single_fd_logs = logs['FastDAC'] if 'FastDAC' in logs.keys() else logs['FastDAC 1']
                fd_group = logs_group.require_group('FastDACs')
                InitLogs.set_fastdac(fd_group, single_fd_logs)
                # TODO: Make work for multiple FastDACs
            if 'FastDAC 2' in logs.keys():
                logger.warning(f'Second FastDAC found in logs, but not implemented making this data nice yet')

            # Temperatures
            if 'Lakeshore' in logs.keys() and 'Temperature' in logs['Lakeshore']:
                temps_dict = logs['Lakeshore']['Temperature']
            elif 'Temperatures' in logs.keys():
                temps_dict = logs['Temperatures']
            else:
                temps_dict = None
            if temps_dict:
                temps_group = logs_group.require_group('Temperatures')
                InitLogs.set_temps(temps_group, temps_dict)

            # Magnets
            # TODO

            # SRS Lock-ins
            # TODO

            # Keithley
            # TODO

            # HP
            # TODO

            # TODO: Anything else?

        except KeyError as e:
            logger.error(f'Error, skipping the rest of making nice Logs: {e}')
            pass





