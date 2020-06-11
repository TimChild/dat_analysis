from src.Core import make_dat_from_exp
# from src.Sandbox import *
import logging

logger = logging.getLogger(__name__)


# h5py.enable_ipython_completer()


if __name__ == '__main__':
    import os

    os.remove(
        r'D:\OneDrive\UBC LAB\My work\Fridge_Measurements_and_Devices\Fridge Measurements with PyDatAnalysis\Jun20\Dat_HDFs\Dat36.h5')
    # dat = JunDatBuilder(36, 'base')
    # dat.init_Data()

    db = make_dat_from_exp(36)
