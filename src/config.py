"""Shared variables, paths and other defaults"""
import os
import pandas as pd

verbose = True
verboselevel = 19

abspath = os.path.abspath('.').split('PyDatAnalysis')[0]
abspath = os.path.join(abspath, 'PyDatAnalysis/src')

ddir = os.path.join(abspath, 'Data/Experiment_Data')
pickledata = os.path.join(abspath, 'Data/Pickles')
plotdir = os.path.join(abspath, 'Data/Plots')
dfdir = os.path.join(abspath, 'Data/DataFrames')


### For how pandas displays in console
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

