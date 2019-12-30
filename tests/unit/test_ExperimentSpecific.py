import unittest
from src.ExampleExperimentSpecific import *
import tests.helpers as th
th.Dirs.set_test_dirs()  # Sets directories to point to fixtures.. Can use th.Dirs.reset... to change back


class Test_make_dat(unittest.TestCase):
    def test_make(self):
        make_dat_standard(2700)
        # Just checking this builds without errors




if __name__ == '__main__':  # This doesn't run when running unit tests with Pycharm...
    unittest.main()
