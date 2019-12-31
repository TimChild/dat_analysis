from unittest import TestCase
from src import Core
import tests.helpers as testhelp
testhelp.Dirs.set_test_dirs()  # Setup directories to point to fixtures


class TestCoreTest(TestCase):
    def test_Normal(self):
        """First test to run"""
        data = [1, 2, 3]
        result = Core.coretest([1, 2, 3])
        self.assertEqual(result, 6)

    def test_Error(self):
        data = None
        with self.assertRaises(TypeError):
            result = Core.coretest(data)


testhelp.Dirs.reset_dirs()  # Reset config directories
