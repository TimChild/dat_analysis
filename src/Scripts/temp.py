# from src.Experiment import *
# from src.DFcode.DatDF import DatDF
# from src.DFcode.SetupDF import SetupDF
from typing import NamedTuple
from src.Sandbox import *


class test(object):
    def __init__(self):
        self.a = 1
        self._b = 2
        self.__c = 3


if __name__ == '__main__':
    t = test()
    d = t.a
    print(d)