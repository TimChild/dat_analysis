"""Shared variables, paths and other defaults"""
import os
import pandas as pd
import src.Jan20Config as ES

verbose = True
verboselevel = 19

datapath = ES.datapath

ddir = os.path.join(datapath, 'Experiment_Data')
pickledata = os.path.join(datapath, 'Pickles')
plotdir = os.path.join(datapath, 'Plots')
dfdir = os.path.join(datapath, 'DataFrames')
dfsetupdirpkl = os.path.join(datapath, 'DataFrames/setup/setup.pkl')
dfsetupdirexcel = os.path.join(datapath, 'DataFrames/setup/setup.xlsx')
dfbackupdir = os.path.join(datapath, 'DataFramesBackups')

commonwavenames = ES.wavenames



### For how pandas displays in console
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

# pd.DataFrame.set_index = protect_data_from_reindex(pd.DataFrame.set_index)  # Protect from deleting columns of data