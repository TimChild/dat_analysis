# from src.Experiment import *
# from src.DFcode.DatDF import DatDF
# from src.DFcode.SetupDF import SetupDF
from typing import NamedTuple
from src.Sandbox import *


def test_kwargs(**kwargs):
    print(**kwargs)
    if kwargs is None:
        print('hi')
    else:
        print(type(kwargs))
    


if __name__ == '__main__':
    hdf = CU.open_hdf5(20, cfg.ddir)
    print(hdf.keys())