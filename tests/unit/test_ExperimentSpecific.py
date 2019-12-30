import unittest
from src import ExampleExperimentSpecific as mod
# from src.ExampleExperimentSpecific import *
import tests.helpers as th
from tests.helpers import change_to_mock_input
from io import StringIO
from unittest.mock import patch
from unittest.mock import MagicMock
import functools

th.Dirs.set_test_dirs()  # Sets directories to point to fixtures.. Can use th.Dirs.reset... to change back


class Test_make_dat(unittest.TestCase):
    inputs = [
        'n',  # replace with excel?
        'y',  # overwrite values?
        'y',
        'y',
        'y',
        'y',
    ]

    @change_to_mock_input(inputs)
    def test_make(self):
        mod.make_dat_standard(2700)


if __name__ == '__main__':  # This doesn't run when running unit tests with Pycharm...
    unittest.main()
