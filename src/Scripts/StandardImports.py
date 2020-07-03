"""Has most imports for normal plotting scripts"""
#  Packages from my code that I use often
from src import CoreUtil as CU
from src import PlottingFunctions as PF
from src.DatObject.Make_Dat import DatHandler
from src.DatObject.DatHDF import DatHDF
from src.DataStandardize.Standardize_Util import wait_for  # Sets a thread waiting for a dat to finish

#  Adds Copy/Paste to figures
import src.AddCopyFig

#  Common packages I use
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import inspect          # Useful for inspect.getSource()  # for storing code in HDF
import lmfit as lm
from typing import List, Tuple, Union, Set, NamedTuple, Dict  # Good for asserting types

#  The Experiment's I'm currently working with. Makes it easier to get to Config/ESI/Fixes
import src.DataStandardize.ExpSpecific.Jun20 as Jun20
import src.DataStandardize.ExpSpecific.Jan20 as Jan20

# Most commonly used functions and classes
get_dat = DatHandler.get_dat
get_dats = DatHandler.get_dats
JunESI = Jun20.JunESI
JanESI = Jan20.JanESI


# Set logging defaults
CU.set_default_logging()
