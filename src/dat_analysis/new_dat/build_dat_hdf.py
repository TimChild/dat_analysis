"""
General functions for building a nicely standardized HDF file from experiment files.
Stuff in here should work quite generally for most experiment files. Really specific stuff (e.g. fixing how speicific dats have been saved should be done outside of this package)


Note:
It is not necessary to use anyhthing from this file to create the standardized HDF files as long as you stick to the correct conventions for the HDF file.
"""
import json
import os
import re
from typing import Optional
import logging
logger = logging.getLogger(__name__)

import h5py

from dat_analysis.dat_object.attributes.logs import InitLogs
from dat_analysis.hdf_file_handler import HDFFileHandler
from dat_analysis.hdf_util import set_attr

from .logs_attr import FastDAC, Temperatures
from ..dat_object.attributes.logs import _dac_logs_to_dict


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

    with HDFFileHandler(new_save_path, 'w') as n:
        match = re.search(r'(\d+)[\._]', os.path.split(exp_data_path)[1])  # Look for number preceding . or _ (e.g. dat1.h5 or dat1_RAW.h5)
        datnum = int(match.groups()[0]) if match else -1
        n.attrs['datnum'] = datnum
        n.attrs['experiment_data_path'] = exp_data_path
        with HDFFileHandler(exp_data_path, 'r') as o:

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
    logs = {}
    try:
        logs = json.loads(sweep_logs_str)
    except json.JSONDecodeError as e:
        logger.error(f'Error, skipping the making nice Logs: {e.msg}')
        logs = None

    if logs:
        try:
            # General Logs
            general_group = logs_group.require_group('General')
            if 'axis_labels' in logs.keys():
                labels = logs.pop('axis_labels')
                general_group.attrs['x_label'] = labels.get('x', None)
                general_group.attrs['y_label'] = labels.get('y', None)
            for k, nk in (  # Usually present, Preferred name
                    ('time_elapsed', None),
                    ('time_completed', None),
                    ('comment', 'comments'),
                    ('resamplingFreq', 'resampling_freq'),
                    ('measureFreq', 'measure_freq'),
            ):
                nk = nk if nk else k
                if k in logs.keys():
                    general_group.attrs[nk] = logs.pop(k)
        except Exception as e:
            logger.error(f'Error making "General" logs')
            raise e

        try:
            # FastDAC
            if 'FastDAC' in logs.keys() or 'FastDAC 1' in logs.keys():
                fd_sweeplogs = logs.pop('FastDAC') if 'FastDAC' in logs.keys() else logs.pop('FastDAC 1')
                fd_log = fd_entry_from_logs(fd_sweeplogs)
                fd_log.save_to_hdf(logs_group, 'FastDAC1')

            for i in range(2, 10):  # Up to 10 fastdacs
                if f'FastDAC {i}' in logs.keys():
                    fd_sweeplogs = logs.pop(f'FastDAC {i}')
                    fd_log = fd_entry_from_logs(fd_sweeplogs)
                    fd_log.save_to_hdf(logs_group, f'FastDAC{i}')

        except Exception as e:
            logger.error(f'Error making "FastDAC" in logs')
            raise e

        try:
            # Temperatures
            if 'Lakeshore' in logs.keys() and 'Temperature' in logs['Lakeshore']:
                temps_dict = logs.pop('Lakeshore')['Temperature']
            elif 'Temperatures' in logs.keys():
                temps_dict = logs.pop('Temperatures')
            else:
                temps_dict = None
            if temps_dict:
                temp_log = temp_entry_from_logs(temps_dict)
                temp_log.save_to_hdf(logs_group, 'Temperatures')
        except Exception as e:
            logger.error(f'Error making "Temperatures" in logs')
            raise e

            # Magnets
            # TODO

            # SRS Lock-ins
            # TODO

            # Keithley
            # TODO

            # HP
            # TODO

        # TODO: Anything else?
        # Left over
        other_group = logs_group.require_group('Other')
        for k, v in logs.items():
            try:
                set_attr(other_group, k, v)
            except Exception:
                pass


def fd_entry_from_logs(fd_log) -> FastDAC:
    visa = fd_log.get('visa_address', None)
    sampling_freq = fd_log.get('SamplingFreq', None)
    measure_freq = fd_log.get('MeasureFreq', None)
    AWG = fd_log.get('AWG', None)

    # Get DACs and ADCs
    dac_vals = {k: fd_log.get(k) for k in fd_log.keys() if re.match(r'DAC\d+{.*}', k)}
    adcs = {k: fd_log.get(k) for k in fd_log.keys() if re.match(r'ADC\d+', k)}

    # Extract names and nums from DACs
    dac_names = [re.search('(?<={).*(?=})', k)[0] for k in dac_vals.keys()]
    dac_nums = [int(re.search('\d+', k)[0]) for k in dac_vals.keys()]

    # Extract nums from ADCs
    adc_nums = [int(re.search('\d+', k)[0]) for k in adcs.keys()]

    # Make sure they are in order
    dac_vals = dict(sorted(zip(dac_nums, dac_vals.values())))
    dac_names = dict(sorted(zip(dac_nums, dac_names)))
    adcs = dict(sorted(zip(adc_nums, adcs.values())))

    # Fill in any missing DAC names (i.e. if not specified in {} then just use DAC#)
    dac_names = {k: n if n else f'DAC{k}' for k, n in dac_names.items()}

    fd = FastDAC(dac_vals=dac_vals, dac_names=dac_names, adcs=adcs, sample_freq=sampling_freq, measure_freq=measure_freq,
                 AWG=AWG,
                 visa_address=visa)
    return fd


def temp_entry_from_logs(tempdict) -> Temperatures:
    tempdata = {'mc': tempdict.get('MC K', None),
                'still': tempdict.get('Still K', None),
                'fourk': tempdict.get('4K Plate K', None),
                'magnet': tempdict.get('Magnet K', None),
                'fiftyk': tempdict.get('50K Plate K', None)}
    temps = Temperatures(**tempdata)
    return temps


def make_aliases_of_standard_data(data_group: h5py.Group):
    """
    Make some soft-links to datasets with more standard names (i.e. i_sense instead of cscurrent, cscurrent_2d, current, etc)
    Args:
        data_group: The top level of the Data group (which contains copies of experiment data, and any other sub groups)

    Returns:
        modifies data_group to include more standard data names in './standard_names/'

    """
    for k in data_group.keys():
        d = data_group.get(k)
        if isinstance(d, h5py.Dataset):

            # Find charge sensing transition data
            if k in ['cscurrent', 'cscurrent_2d']:
                standard = data_group.require_group('standard')
                standard['i_sense'] = d










