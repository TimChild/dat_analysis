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
from dat_analysis.dat.build_dat_hdf import check_hdf_meets_requirements, default_exp_to_hdf, default_sort_sweeplogs, make_aliases_of_standard_data
from dat_analysis.hdf_file_handler import HDFFileHandler  # Use this to open HDF files to avoid OS Errors


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
    if re.search(r'dat(\d+)', experiment_data_path.split()[-1]):
        datnum = int(re.search(r'dat(\d+)', experiment_data_path.split()[-1]).groups()[0])

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
    if user_name.lower() == 'tim':
        # Possibly decide to not even use the default exp_to_hdf function
        # pass -- Don't need this yet

        # Move over most stuff in a reasonably good way
        default_exp_to_hdf(exp_data_path=experiment_data_path, new_save_path=DatHDF_save_location)

        # Fix other things than need fixing after the initial move
        if host_name == 'qdev-xld':
            if experiment == '202208_KondoConductanceDots':
                if datnum == 305:
                    # Replace a few bad datapoints due to touching something with 0
                    import numpy as np
                    with HDFFileHandler(DatHDF_save_location, 'r+') as f:
                        data = f['Data']['current_2d']
                        bad = np.argwhere(np.abs(data) > 0.6)
                        for y, x in bad:
                            f['Data']['current_2d'][y, x] = 0
                        f['Data']['current_2d'].attrs['loading_modifications'] = 'replaced a few bad values (due to ' \
                                                                                 'touching something) with 0 '

        # Make the standard data names available in Data/standard/...
        with HDFFileHandler(DatHDF_save_location, 'r+') as f:
            make_aliases_of_standard_data(f['Data'])
        return True
    else:
        return default_exp_to_hdf(exp_data_path=experiment_data_path, new_save_path=DatHDF_save_location)


def example_converter(exp_path, save_loc):
    """Just an example, anything else could be done in here to make the next DatHDF file from exp file"""
    return default_exp_to_hdf(exp_data_path=exp_path, new_save_path=save_loc)



if __name__ == '__main__':
    from dat_analysis import get_dat
    dat = get_dat(305)