"""
General utility functions related to the new simpler dat HDF interface
"""
import os
import toml
import json
import numpy as np
from typing import Optional
import logging
from ..core_util import get_full_path
logger = logging.getLogger(__name__)

config_path = os.environ.get('DatAnalysisConfig', None)

if config_path is None:
    msg = f'Warning: No "DatAnalysisConfig" environment variable found, there will be no default config. \n\n'\
          f'For first setup, you need to provide a path to a "config.toml" file (note: may have to change\n '\
          f'permissions depending on how you python is installed). If the file empty or even non-existent (\n'\
          f'as long as a file can be created there), a default template will be made the next time you run \n'\
          f'the program. \n' \
          f'You may have to restart your python environment (e.g. jupyter, pycharm, etc).\n\n' \
          f'Note: This warning can be ignored in certain cases'
    logger.warning(msg)


def default_config():
    """Makes a default .toml config file for dat_analysis"""
    config = {'loading': {
        'path_to_measurement_data': '',
        'path_to_save_directory': '',
        # 'default_host_name': '',
        # 'default_user_name': '',
        # 'default_experiment_name': '',
        'path_to_python_load_file': '',
    }}
    return config


def get_local_config(path: Optional[str] = None) -> dict:
    """Get the configuration file saved on the machine (i.e. default place to store HDF files, default place to find
    experiment files, etc)"""
    if not path:
        path = config_path

    if path:
        if not os.path.exists(path) or (os.path.exists(path) and os.path.getsize(path) == 0):
            with open(path, 'w') as f:
                toml.dump(default_config(), f)
    else:
        raise RuntimeError(f'Path to DatAnalysisConfig not found. Please set "DatAnalysisConfig" environment variable to point to a "config.toml" file, or wherever you want that file to exist')

    config = toml.load(path)
    for k in config['loading'].keys():
        if k in ['path_to_measurement_data', 'path_to_save_directory', 'path_to_python_load_file']:
            if p := config['loading'][k]:
                config['loading'][k] = get_full_path(p)
    return config


class NpEncoder(json.JSONEncoder):
    """
    Allows Json to dump things that have numpy numbers in

    Examples:
        json.dumps(<object_with_numpy_numbers>, cls=NpEncoder)
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
