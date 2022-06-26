import os
import toml
from dat_analysis.new_dat.new_dat_util import get_local_config, default_config


def setup_test_config():
    """Sets up the config.toml file for tests and makes the default config the test one (should be run at the beginning
    of most test files)"""
    import dat_analysis.new_dat.new_dat_util as ndu
    tests_dir = os.sep.join(os.path.normpath(__file__).split(os.sep)[:-2])
    path = os.path.join(tests_dir, os.path.normpath('fixtures/config.toml'))
    local_only_path = os.path.join(tests_dir, os.path.normpath('fixtures/.config.toml'))  # Gets overwritten with absolute filepaths

    # Make some paths absolute (so tests can run on any system)
    # path_to_tests = os.path.normpath(__file__.split()[0].split()[0])  # e.g. Should contain fixtures, Outputs etc
    config = get_local_config(path)
    for key in ['path_to_measurement_data', 'path_to_save_directory', 'path_to_python_load_file']:
        config['loading'][key] = os.path.join(tests_dir, os.path.normpath(config['loading'][key]))
    with open(local_only_path, 'w') as f:
        toml.dump(config, f)

    # So any default loading of config will also point to this config.toml
    ndu.config_path = os.path.abspath(local_only_path)

    # May as well return it
    return config
