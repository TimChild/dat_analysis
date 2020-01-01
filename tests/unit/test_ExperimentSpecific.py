import unittest
from src import ExampleExperimentSpecific as EES
import tests.helpers as th


th.Dirs.set_test_dirs()  # Sets directories to point to fixtures.. Can use th.Dirs.reset... to change back
th.setverbose(False)


class Test_make_dat(unittest.TestCase):
    def test_make(self):
        inputs = [
            'n',  # replace with excel?
            'y',  # overwrite values?
            'y',
            'y',
            'y',
            'y',
        ]

        with unittest.mock.patch('builtins.input', side_effect=th.simple_mock_input(inputs)) as mock:
            EES.make_dat_standard(2700)

