"""Shared variables, paths and other defaults"""
import pandas as pd
import logging
from sys import platform

if platform == "win32":
    main_data_path = 'D:\\OneDrive\\UBC LAB\\My work\\Fridge_Measurements_and_Devices\\Fridge Measurements with PyDatAnalysis'
elif platform == "darwin":
    main_data_path = "/Users/owensheekey/Nextcloud/Shared/measurement-data/Owen"



# Set other defaults
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
# pd.DataFrame.set_index = protect_data_from_reindex(pd.DataFrame.set_index)  # Protect from deleting columns of data

PF_binning = True  # For auto applying binning to plots so no more than num_points_per_row shown per row
PF_num_points_per_row = 1000  # Max num_points to show per row of plot

FIT_NUM_BINS = 1000

logging.basicConfig(level=logging.INFO)  # Set logging to print INFO level events logged.
# (each logger still needs level set to INFO)

# Shared variables to be passed around by functions elsewhere
warning = None
yes_to_all = False
