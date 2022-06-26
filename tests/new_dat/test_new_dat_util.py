from unittest import TestCase
import toml
import os
from dat_analysis.new_dat.new_dat_util import get_local_config, default_config

test_config_path = 'test_config.toml'


class Test(TestCase):
    def test_get_environ_config(self):
        p = 'test_config.toml'
        os.environ['TestDatAnalysisConfig'] = p
        config_path = os.environ.get('TestDatAnalysisConfig', None)
        self.assertEqual(p, config_path)



    def test_get_local_config(self):
        config = get_local_config(path=test_config_path)
        self.assertIsInstance(config, dict)
        self.assertTrue('path_to_measurement_data' in config.keys())
        self.assertTrue('path_to_save_directory' in config.keys())

    def test_default_config(self):
        config = default_config()
        config_str = toml.dumps(config)
        config_load = toml.loads(config_str)
        self.assertEqual(config, config_load)
