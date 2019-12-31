import unittest
from src import ExampleExperimentSpecific as EES
import tests.helpers as th
from tests.helpers import change_to_mock_input


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

        @change_to_mock_input(EES, inputs)  # Changes module to have fake inputs
        def runtest():  # Just testing for no errors
            EES.make_dat_standard(2700)
        runtest()
