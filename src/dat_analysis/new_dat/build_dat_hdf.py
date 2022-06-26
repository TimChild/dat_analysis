"""
General functions for building a nicely standardized HDF file from experiment files.
Stuff in here should work quite generally for most experiment files. Really specific stuff (e.g. fixing how speicific dats have been saved should be done outside of this package)


Note:
It is not necessary to use anyhthing from this file to create the standardized HDF files as long as you stick to the correct conventions for the HDF file.
"""
import json
import os
from typing import Optional
import logging
logger = logging.getLogger(__name__)

import h5py

from dat_analysis.dat_object.attributes.logs import InitLogs
from dat_analysis.hdf_file_handler import HDFFileHandler


def check_hdf_meets_requirements(path: str):
    """Check the hdf_path points to an HDF file that contains expected groups/attrs"""
    passes = True
    messages = []
    with HDFFileHandler(path, 'r') as f:
        # Check for some standard attrs that should exist
        if 'experiment_data_path' not in f.attrs.keys():
            passes = False
            messages.append("Did not find 'experiment_data_path' as a top level attr")

        # Check for the standard groups that should exist
        keys = f.keys()
        for k in ['Logs', 'Data']:
            if k not in keys:
                passes = False
                messages.append(f'Did not find group {k} in top level')

    message = '\n'.join(messages)
    return passes, message


def default_exp_to_hdf(exp_data_path: str, new_save_path: str):
    """Copy the relevant information from experiment Dat into standardized DatHDF

    Note: May be necessary to copy more stuff after depending on the state of the experiment hdf file (i.e. bad jsons
    or missing metadata)
    """
    if os.path.exists(new_save_path):
        raise FileExistsError(f'File already exists at {new_save_path}')

    with HDFFileHandler(exp_data_path, 'r') as o:
        with HDFFileHandler(new_save_path, 'w') as n:
            n.attrs['experiment_data_path'] = exp_data_path

            data_group = n.require_group('Data')
            logs_group = n.require_group('Logs')
            for k in o.keys():
                if isinstance(o[k], h5py.Dataset):
                    data_group[k] = o[k][:]

            if 'metadata' in o.keys():
                    if 'sweep_logs' in o['metadata'].attrs.keys():
                        sweeplogs_str = o['metadata'].attrs['sweep_logs']
                        logs_group.attrs['sweep_logs_string'] = sweeplogs_str  # Make the full recorded string available
                        default_sort_sweeplogs(logs_group, sweeplogs_str)
                    if 'ScanVars' in o['metadata'].attrs.keys():
                        scanvars_str = o['metadata'].attrs['ScanVars']
                        logs_group.attrs['scan_vars_string'] = scanvars_str  # Make the full recorded string available
                        # TODO: sort the scanvars into something nice
                    # TODO: Also config?


def default_sort_sweeplogs(logs_group: h5py.Group, sweep_logs_str: str):
    """Convert the sweep_logs string into standard attributes in logs_group

    This should work for most recent Dats (~2021+) excluding those that were saved with issues. This function should
    NOT be altered to account for temporary issues with Dats (to prevent this growing into a huge mess). Instead,
    """

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
                InitLogs.set_fastdac(logs_group, single_fd_logs)
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
                InitLogs.set_temps(logs_group, temps_dict)

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
