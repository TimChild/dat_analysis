"""
2022 -- Way too out of date...
"""
import os

path_replace = None


instruments = {'srs': 'srs830', 'dmm': 'hp34401a', 'dac': 'babydac', 'fastdac': 'fastdac', 'magnet':'ls625', 'fridge':'ls370'}
instrument_num = {'srs': 4, 'dmm': 1, 'dac': 16, 'fastdac': 4, 'magnet': 3}


dat_types_list = ['i_sense', 'entropy', 'transition', 'pinch', 'dot tuning']

### Path to all Data (e.g. dats, dataframes, pickles etc). Hopefully this will allow moving out of project without
#  losing access to everything
dir_name = 'Sept19'


wavenames = ['i_sense', 'FastScan'] + [f'FastScanCh{i}' for i in range(4)]  # + [f'fd{i}adc' for i in range(4)]
raw_wavenames = []

i_sense_keys = ['FastScanCh0_2D', 'FastScan_2D', 'fd_0adc']
entropy_x_keys = ['FastScanCh1_2D', 'fd_1adc']
entropy_y_keys = ['FastScanCh2_2D', 'fd_2adc']


def json_sub_getter(datnum):  # So that I can send back datnum specific json_subs
    if datnum >= 1300:  # 1300 is just a guess as to where I fixed the dacnames issue, may need to be adjusted
        return [(', "range":, "resolution":', ""), (":\+", ':')]
    elif datnum < 1300:
        return [(', "range":, "resolution":', ""), (":\+", ':'), ('"CH0name".*"com_port"', '"com_port"')]


json_subs = json_sub_getter

DC_HQPC_current_bias_resistance = 10e6  # Resistor inline with DC bias for HQPC
