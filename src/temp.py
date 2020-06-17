from src.Scripts.StandardImports import *
import abc

class TestBaseClass(abc.ABC):
    def __init__(self):
        self.a = 1
        self.b = 2

    @abc.abstractmethod
    def test_method(self, val=None):
        if val is None:
            val = 10
        return val

class TestSubClass(TestBaseClass):

    def test_method(self, val=None):
        val = super().test_method(val = val)
        print(val)



if __name__ == '__main__':
    a = TestSubClass()
    a.test_method(val=None)
    a.test_method(val=5)

