"""Shared variables, paths and other defaults"""
import os
import pandas as pd
import win32com.client
from typing import Union
import src.Configs.Jan20Config as ES

# Define Config specific functions
def set_paths_from_config(main_data_path, config=ES):
    """
    Sets paths for all experiment files. I.e. paths for ddir, pickledata, plotdir, dfdir, dfsetupdirpkl, dfsetupdirexcel, dfbackupdir
    @param datapath: Path to main folder which contains DataFrames, Pickles, Experiment_Data etc
    """
    datapath = os.path.join(main_data_path, config.dir_name)
    global ddir, pickledata, plotdir, dfdir, dfsetupdir, dfsetupdirpkl, dfsetupdirexcel, dfbackupdir
    ddir = os.path.join(datapath, 'Experiment_Data')
    pickledata = os.path.join(datapath, 'Pickles')
    plotdir = os.path.join(datapath, 'Plots')
    dfdir = os.path.join(datapath, 'DataFrames')
    dfsetupdir = os.path.join(datapath, 'DataFrames/setup/')
    # dfsetupdirpkl = os.path.join(datapath, 'DataFrames/setup/setup.pkl')
    # dfsetupdirexcel = os.path.join(datapath, 'DataFrames/setup/setup.xlsx')
    dfbackupdir = os.path.join(datapath, 'DataFramesBackups')

    # #  In case paths have been replaced with shortcuts (i.e. archived experiments)
    # ddir = get_correct_path(ddir)
    # pickledata = get_correct_path(pickledata)
    # dfbackupdir = get_correct_path(dfbackupdir)

#  TODO: These are in CoreUtil as well, just need to figure out how to work with dependencies
def _get_path_from_shortcut(path):
    shell = win32com.client.Dispatch("WScript.Shell")
    path = path + '.lnk'  # If it's a shortcut instead of a folder it will appear as a .lnk file
    if os.path.isfile(path) is True:
        shortcut = shell.CreateShortCut(path)
    else:
        raise ValueError(f'Path "{path}" is not a shortcut link')
    return shortcut.TargetPath

#  TODO: Same as _get_path.... note above
def get_correct_path(path):
    """Just returns the path if it points to a directory, otherwise checks to see if the path is a shortcut and if so
    returns the full path of the target location"""
    if os.path.isdir(path) is False:
        path = _get_path_from_shortcut(path)
    return path


def set_jsonsubs(config=ES):
    """
    Sets json_subs variable to equal list of json_subs from given config file or defaults to ES config

    @param config: Option to pass in a different config module from which to grab json_subs
    @type config: module
    """
    global json_subs
    try:  # Get list of json substitutions that potentially need to be made (fixing broken jsons)
        json_subs = config.json_subs
        assert type(json_subs) == list
        if len(json_subs) > 0:
            assert type(json_subs[0]) == tuple  # Looking for (sub, repl) pairs
    except AttributeError:
        json_subs = []


def set_dattypes(config=ES):
    """
    Sets dat_types variable to expected dat types from given config file or defaults to ES config

    @param config: Option to pass in different config module from which to grab json_subs
    @type config: module
    """
    global dat_types
    try:  # Get list of dat_types
        dat_types = config.dat_types_list
        assert type(dat_types) == list
    except AttributeError:
        dat_types = None


def set_wavenames(config=ES):
    """
    Sets common_wavenames and common_raw_wavenames variables from given config file or defaults to ES config

    @param config: Option to pass in different config module from which to grab expected wavenames
    @type config: module
    """
    global common_wavenames, common_raw_wavenames
    common_wavenames = config.wavenames
    common_raw_wavenames = config.raw_wavenames


def set_other_variables(config=ES):
    """
    Sets any other variables which may be defined in a given config file or defaults to using ES config

    @param config: Option to pass in different config module from which to grab any other variables
    @type config: module
    """
    global DC_current_bias_resistance
    try:
        DC_current_bias_resistance = config.DC_HQPC_current_bias_resistance
    except:  # TODO: only need to except if variable is missing.
        pass


def set_path_replace(config=ES):
    """
    In case path to files has changed at some point (i.e. moved whole experiment folder to different directory)

    @param config: Option to pass in different config module from which to grab any other variables
    @type config: module
    """

    global path_replace
    try:
        path_replace = config.path_replace
    except:
        path_replace = None


def set_all_for_config(config=ES, folder_containing_experiment=None):
    global current_config, current_folder_containing_experiment
    if folder_containing_experiment is None:
        global main_data_path
        folder_containing_experiment = main_data_path
    set_paths_from_config(folder_containing_experiment, config)
    set_path_replace(config)
    set_jsonsubs(config)
    set_dattypes(config)
    set_wavenames(config)
    set_other_variables(config)
    current_config = config
    current_folder_containing_experiment = folder_containing_experiment


# Begin Initialization of Main_Config ################################################################
# Initialize global variables
verbose = False  # Not really used
verboselevel = 19  # Not really used
yes_to_all = False  # to make CoreUtil.Option_input() answer 'yes' automatically to yes or no question.
warning = None  # Used for functions to pass back optional warnings

# Initialize other global variables to None for typing purposes
path_replace = None
ddir = None
pickledata = None
plotdir = None
dfdir = None
dfsetupdir = None
dfsetupdirpkl = None
dfsetupdirexcel = None
dfbackupdir = None
dat_types = None
json_subs = None
common_wavenames = None
common_raw_wavenames = None
DC_current_bias_resistance = None

#  Define main data path (i.e. where all experiments are stored)
main_data_path = 'D:\\OneDrive\\UBC LAB\\My work\\Fridge_Measurements_and_Devices\\Fridge Measurements with PyDatAnalysis'
current_config = ES  # For storing which config is currently being used
current_folder_containing_experiment = main_data_path  # i.e. Folder which contains several whole experiments

# Set all other global variables for ES config file by default and uses main_data_path by default
set_all_for_config()

# Set other defaults
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
# pd.DataFrame.set_index = protect_data_from_reindex(pd.DataFrame.set_index)  # Protect from deleting columns of data
