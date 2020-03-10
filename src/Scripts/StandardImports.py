"""Has most imports for normal plotting scripts"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import src.CoreUtil as CU
import src.DatCode.Datutil as DU
import src.DFcode.DatDF as DF
from src.DFcode.DatDF import update_save
import src.DFcode.SetupDF as SF
import src.PlottingFunctions as PF
import src.Configs.Main_Config as cfg
from src.Core import make_dat_standard
from src.Core import make_dats
import src.Core as C
import src.DatCode.Dats as Dats
from src.PlottingFunctions import ax_setup

import src.AddCopyFig  # I think this needs to be imported only once!!

cfg.yes_to_all = True
datdf = DF.DatDF()
cfg.yes_to_all = False
setupdf = SF.SetupDF()
