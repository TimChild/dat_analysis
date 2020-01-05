import unittest
from src import Sep19ExperimentSpecific as EES
import tests.helpers as th


th.Dirs.set_test_dirs()  # Sets directories to point to fixtures.. Can use th.Dirs.reset... to change back
th.setverbose(False)


class Test_make_dat(unittest.TestCase):
    def test_make(self):
        pass

