import os
instruments = {'srs': 'srs830', 'dmm': 'hp34401a', 'dac': 'babydac', 'fastdac': 'fastdac', 'magnet':'ls625', 'fridge':'ls370'}
instrument_num = {'srs': 3, 'dmm': 1, 'dac': 16, 'fastdac': 8, 'magnet': 3}


dat_types_list = ['none', 'i_sense', 'entropy', 'transition', 'pinch', 'dot tuning']

### Path to all Data (e.g. dats, dataframes, pickles etc). Hopefully this will allow moving out of project without
#  losing access to everything
abspath = os.path.abspath('..').split('PyDatAnalysis')[0]
dir_name = 'Nik_entropy_v2'  # Name of folder inside main data folder specified by src.config.main_data_path

wavenames = ['x_array', 'y_array', 'i_sense', 'entx', 'enty']

i_sense_keys = ['i_sense']
entropy_x_keys = ['entx']
entropy_y_keys = ['enty']

jsonsubs = [('"comment": "{"gpib_address":4, "units":"VOLT", "range":.1.000000E.0., "resolution":...000000E-0.}"',
                '"comment": "replaced to fix json"'),
            (":\+", ':')]
