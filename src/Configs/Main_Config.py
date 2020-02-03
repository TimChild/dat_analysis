"""Shared variables, paths and other defaults"""
import os
import pandas as pd
import src.Configs.Jan20Config as ES

verbose = False
verboselevel = 19

try:  # Get list of json substitutions that potentially need to be made (fixing broken jsons)
    jsonsubs = ES.jsonsubs
    assert type(jsonsubs) == list
    assert type(jsonsubs[0]) == tuple
except AttributeError:
    jsonsubs = None

main_data_path = 'D:\\OneDrive\\UBC LAB\\My work\\Fridge Measurements with PyDatAnalysis'
datapath = os.path.join(main_data_path, ES.dir_name)

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