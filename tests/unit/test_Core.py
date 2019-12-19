import unittest
from src import Core
from tests.helpers import change_to_test_dir


class TestCoreTest(unittest.TestCase):
    def test_Normal(self):
        """First test to run"""
        data = [1, 2, 3]
        result = Core.coretest([1, 2, 3])
        self.assertEqual(result, 6)

    def test_Error(self):
        data = None
        with self.assertRaises(TypeError):
            result = Core.coretest(data)


class TestCoreDat(unittest.TestCase):
    pass


unittest.main = change_to_test_dir(unittest.main)  # Wraps with temporary directory change
if __name__ == '__main__':
    unittest.main()

