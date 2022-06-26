"""
General utility functions related to the new simpler dat HDF interface
"""
import os
import toml
from typing import Optional
import logging
logger = logging.getLogger(__name__)


config_path = os.environ.get('DatAnalysisConfig', None)
if config_path is None:
    msg = f'No "DatAnalysisConfig" environment variable found, there will be no default config. \n\n'\
          f'For first setup, you need to provide a path to a "config.toml" file (note: may have to change\n '\
          f'permissions depending on how you python is installed). If the file empty or even non-existent (\n'\
          f'as long as a file can be created there), a default template will be made the next time you run \n'\
          f'the program. \n' \
          f'You may have to restart your python environment (e.g. jupyter, pycharm, etc).\n\n'
    logger.warning(msg)
    print(msg)


def default_config():
    """Makes a default .toml config file for dat_analysis"""
    config = {'loading': {
        'path_to_measurement_data': '',
        'path_to_save_directory': '',
        'current_experiment_path': '',
        'path_to_python_load_file': '',
    }}
    return config


def get_local_config(path: Optional[str] = None) -> dict:
    """Get the configuration file saved on the machine (i.e. default place to store HDF files, default place to find
    experiment files, etc)"""
    if not path:
        path = config_path

    if path and not os.path.exists(path):
        with open(path, 'w') as f:
            toml.dump(default_config(), f)

    config = toml.load(path)
    return config

