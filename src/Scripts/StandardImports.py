"""Has most imports for normal plotting scripts"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import src.CoreUtil as CU
import src.DFcode.DatDF as DF
import src.DFcode.SetupDF as SF
import src.PlottingFunctions as PF
import src.Configs.Main_Config as cfg
from src.Core import make_dat_standard
import src.Core as C
cfg.yes_to_all = True
datdf = DF.DatDF()
cfg.yes_to_all = False
setupdf = SF.SetupDF()
