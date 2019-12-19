"""Shared variables, paths and other defaults"""

import os
import pandas as pd

ddir = os.path.join('src/Data/Experiment_Data')
pickledata = os.path.join('src/Data/Pickles')
plotdir = os.path.join('src/Data/Plots')
dfdir = os.path.join('src/Data/DataFrames')


### For how pandas displays in console
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)