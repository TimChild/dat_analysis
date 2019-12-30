import unittest
from src.ExampleExperimentSpecific import *
from tests.helpers import change_to_test_dir


class Test_make_dat(unittest.TestCase):
    def test_make(self):
        make_dat_standard(2700)
        # Just checking this builds without errors


unittest.main = change_to_test_dir(unittest.main)  # Wraps with temporary directory change
if __name__ == '__main__':
    unittest.main()
