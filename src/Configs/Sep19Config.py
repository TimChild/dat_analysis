import os
instruments = {'srs': 'srs830', 'dmm': 'hp34401a', 'dac': 'babydac', 'fastdac': 'fastdac', 'magnet':'ls625', 'fridge':'ls370'}
instrument_num = {'srs': 4, 'dmm': 1, 'dac': 16, 'fastdac': 4, 'magnet': 3}


dat_types_list = ['i_sense', 'entropy', 'transition', 'pinch', 'dot tuning']

### Path to all Data (e.g. dats, dataframes, pickles etc). Hopefully this will allow moving out of project without
#  losing access to everything
dir_name = 'Sept19 - Not complete'


wavenames = ['i_sense', 'FastScan'] + [f'FastScanCh{i}' for i in range(4)] # + [f'fd{i}adc' for i in range(4)]

i_sense_keys = ['FastScanCh0_2D', 'FastScan2D', 'fd_0adc']
entropy_x_keys = ['FastScanCh1_2D', 'fd_1adc']
entropy_y_keys = ['FastScanCh2_2D', 'fd_2adc']

json_subs = [(', "range":, "resolution":', ""), (":\+", ':')]#, ('"CH0name".*"com_port"', '"com_port"')]
