from unittest import TestCase
import toml
import os
from dat_analysis.new_dat.new_dat_util import get_local_config, default_config

# Not permanent, free to be deleted in tests etc
test_config_output_path = '../Outputs/new_dat/test_config.toml'
test_config_fixture_path  = '../fixtures/test_measurement_data_dir/'

class Test(TestCase):
    def test_get_environ_config(self):
        """Just check that setting and getting environment variables works on whatever operating system this is"""
        p = test_config_output_path
        os.environ['TestDatAnalysisConfig'] = p
        config_path = os.environ.get('TestDatAnalysisConfig', None)
        self.assertEqual(p, config_path)

    def test_get_local_config(self):
        """Should return a valid config toml dict (also checking the Test folders are working right)"""
        for config_path in [test_config_output_path, test_config_fixture_path]:
            config = get_local_config(path=config_path)
            self.assertIsInstance(config, dict)
            self.assertTrue('loading' in config.keys())
            loading = config['loading']
            self.assertTrue('path_to_measurement_data' in loading.keys())
            self.assertTrue('path_to_save_directory' in loading.keys())

    def test_default_config(self):
        config = default_config()
        config_str = toml.dumps(config)
        config_load = toml.loads(config_str)
        self.assertEqual(config, config_load)

    def test_setting_test_config_location(self):
        import dat_analysis.new_dat.new_dat_util as ndu

        # Attempt to overwrite the default location in the module
        ndu.config_path = test_config_output_path

        # Make a new config and save at that location
        test_config = default_config()
        test_config['loading']['path_to_measurement_data'] = test_config_output_path
        with open(test_config_output_path, 'w') as f:
            toml.dump(test_config, f)

        # Load from hopefully overwritten config_path
        config = ndu.get_local_config()

        # Confirm
        self.assertEqual(test_config, config)
        
