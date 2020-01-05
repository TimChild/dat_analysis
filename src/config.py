"""Shared variables, paths and other defaults"""
import os
import pandas as pd
# from src.DFcode.DFutil import protect_data_from_reindex

verbose = True
verboselevel = 19

abspath = os.path.abspath('.').split('PyDatAnalysis')[0]
abspath = os.path.join(abspath, 'PyDatAnalysis/src')

ddir = os.path.join(abspath, 'Data/Experiment_Data')
pickledata = os.path.join(abspath, 'Data/Pickles')
plotdir = os.path.join(abspath, 'Data/Plots')
dfdir = os.path.join(abspath, 'Data/DataFrames')
dfsetupdirpkl = os.path.join(abspath, 'Data/DataFrames/setup/setup.pkl')
dfsetupdirexcel = os.path.join(abspath, 'Data/DataFrames/setup/setup.xlsx')

commonwavenames = ['i_sense', 'FastScan'] + [f'FastScanCh{i}' for i in range(4)] # + [f'fd{i}adc' for i in range(4)]



### For how pandas displays in console
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

# pd.DataFrame.set_index = protect_data_from_reindex(pd.DataFrame.set_index)  # Protect from deleting columns of data