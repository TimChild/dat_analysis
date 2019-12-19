import unittest
from src.ExampleExperimentSpecific import *

class Test_make_dat(unittest.TestCase):
    def test_make(self):
        make_dat_standard(2700)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
