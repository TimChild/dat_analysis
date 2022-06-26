"""
This is where the functions that convert from experiment hdf file to standardized DatHDF file should live.

I.e. not really specific to the package, just required to convert from however data is stored at experiment time, to
a standard HDF format that the dat_analysis package can work with.

This file can live anywhere, where the path should be specified in the config.toml file


"""
# Some possibly useful things to help with converting from datXX.h5 to standardized DatHDF.h5 file
import os
import re
import datetime
from dat_analysis.new_dat.build_dat_hdf import check_hdf_meets_requirements, default_exp_to_hdf, default_sort_sweeplogs
from dat_analysis.hdf_util import HDFFileHandler  # Use this to open HDF files to avoid OS Errors


def create_standard_hdf(experiment_data_path: str, DatHDF_save_location, **kwargs):
    """
    This function will be called in order to initialize the standard HDF from the experiment HDF file if it does not
    already exist (or is being overwritten).

    Args:
        experiment_data_path ():  Where the data to be converted is stored
        DatHDF_save_location ():  Where the standardized DatHDF.h5 file should be stored
        **kwargs ():  Any other key word arguments that are passed along (i.e. in case you want to add other arguments)

    Returns:

    """
    # Can figure out some things about the file just from the filename, this might be useful for deciding what needs to
    # be done to make the standard HDF file

    # Very likely that the filename is datXX.h5 or datXX_RAW.h5
    datnum = -1
    if re.match(r'dat(\d+)', experiment_data_path.split()[-1]):
        datnum = re.match(r'dat(\d+)', experiment_data_path.split()[-1]).groups()[0]

    # If pulling from server, the data_path is likely in the form X:/host_name/user_name/experiment_name(/...)/datXX.h5
    host_name, user_name, experiment = '', '', ''
    if len(os.path.normpath(experiment_data_path).split(os.sep)) > 4:
        path_components = os.path.normpath(experiment_data_path).split(os.sep)
        host_name, user_name = path_components[1:3]
        experiment = '/'.join(path_components[3:-1])  # In case more than one level of depth

    # Quite likely to have the time the file was saved in sweeplogs
    time_completed = None
    try:
        with HDFFileHandler(experiment_data_path, 'r') as f:
            if 'metadata' in f.keys() and 'sweep_logs' in f['metadata'].attrs.keys():
                time_completed = datetime.datetime.strptime(re.search(r'"time_completed": "(.+)"', f['metadata'].attrs['sweep_logs'])[0], '%B %d, %Y %H:%M:%S')
    except (ValueError, TypeError) as e:  # VE if datetime is bad, TE if no datetime found
        pass

    # Examples of how the converter might be chosen
    if host_name == 'qdev-xld' and user_name == 'Tim' and experiment == '202206_TestCondKondoQPC' and datnum > 0:
        return example_converter(experiment_data_path, DatHDF_save_location)
    elif time_completed and time_completed > datetime.datetime(2022, 4, 11):
        return example_converter(experiment_data_path, DatHDF_save_location)
    else:
        return default_exp_to_hdf(exp_data_path=experiment_data_path, new_save_path=DatHDF_save_location)


def example_converter(exp_path, save_loc):
    """Just an example, anything else could be done in here to make the next DatHDF file from exp file"""
    return default_exp_to_hdf(exp_data_path=exp_path, new_save_path=save_loc)