import os
instruments = {'srs': 'srs830', 'dmm': 'hp34401a', 'dac': 'babydac', 'fastdac': 'fastdac', 'magnet':'ls625', 'fridge':'ls370'}
instrument_num = {'srs': 3, 'dmm': 1, 'dac': 16, 'fastdac': 8, 'magnet': 3}


dat_types_list = ['i_sense', 'entropy', 'transition', 'pinch', 'dot tuning']

### Path to all Data (e.g. dats, dataframes, pickles etc). Hopefully this will allow moving out of project without
#  losing access to everything
abspath = os.path.abspath('.').split('PyDatAnalysis')[0]
datapath = os.path.join(abspath, 'PyDatAnalysis/src/Data')  # For now just path to inside of project

wavenames = ['x_array', 'y_array', 'i_sense', 'entx', 'enty']

i_sense_keys = ['i_sense']
entropy_x_keys = ['entx']
entropy_y_keys = ['enty']