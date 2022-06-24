"""
General utility functions related to the new simpler dat HDF interface
"""
import os
import toml
from typing import Optional

from ..core_util import get_project_root

root = get_project_root()
config_path = os.path.join(root, 'config.toml')


def default_config():
    """Makes a default .toml config file for dat_analysis"""
    config = {
        'path_to_measurement_data': '',
        'path_to_save_directory': '',
        'current_experiment_path': '',
    }
    return config


def get_local_config(path: Optional[str] = None) -> dict:
    """Get the configuration file saved on the machine (i.e. default place to store HDF files, default place to find
    experiment files, etc)"""
    if not path:
        path = config_path

    if not os.path.exists(path):
        with open(path, 'w') as f:
            toml.dump(default_config(), f)

    config = toml.load(path)
    return config

