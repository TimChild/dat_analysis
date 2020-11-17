from unittest import TestCase
from src.DataStandardize.BaseClasses import ExpConfigBase
from tests.helpers import get_testing_ExpConfig, init_testing_dat

output_dir = '/Outputs/ExpConfig'

ExpConfig = get_testing_ExpConfig()

dat = init_testing_dat(datnum=9111, output_directory=output_dir)

class Test_ExpConfig(TestCase):
    pass