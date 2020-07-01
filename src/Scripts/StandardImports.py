"""Has most imports for normal plotting scripts"""
from src import CoreUtil as CU
from src import PlottingFunctions as PF
from src.DatObject.Make_Dat import DatHandler
get_dat = DatHandler.get_dat

import src.AddCopyFig
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lmfit as lm

CU.set_default_logging()

# cfg.yes_to_all = True
# datdf = DF.DatDF()
# cfg.yes_to_all = False
# setupdf = SF.SetupDF()


if __name__ == '__main__':
    pass