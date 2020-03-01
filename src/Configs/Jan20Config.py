import os
instruments = {'srs': 'srs830', 'dmm': 'hp34401a', 'dac': 'babydac', 'fastdac': 'fastdac', 'magnet':'ls625', 'fridge':'ls370'}
instrument_num = {'srs': 3, 'dmm': 1, 'dac': 16, 'fastdac': 8, 'magnet': 3}

dat_types_list = ['none', 'i_sense', 'entropy', 'transition', 'pinch', 'dot tuning', 'dcbias']

### Path to all Data (e.g. dats, dataframes, pickles etc). Hopefully this will allow moving out of project without
#  losing access to everything
abspath = os.path.abspath('..').split('PyDatAnalysis')[0]
dir_name = 'Nik_entropy_v2'  # Name of folder inside main data folder specified by src.config.main_data_path

wavenames = ['x_array', 'y_array', 'i_sense', 'entx', 'enty']
raw_wavenames = [f'ADC{i:d}' for i in range(4)] + [f'ADC{i:d}_2d' for i in range(4)] + ['g1x', 'g1y', 'g2x', 'g2y']

i_sense_keys = ['i_sense', 'cscurrent', 'cscurrent_2d']
entropy_x_keys = ['entx', 'entropy_x_2d', 'entropy_x']
entropy_y_keys = ['enty', 'entropy_y_2d', 'entropy_y']

jsonsubs = [('"comment": "{"gpib_address":4, "units":"VOLT", "range":.1.000000E.0., "resolution":...000000E-0.}"',
                '"comment": "replaced to fix json"'),
            (":\+", ':'),
            ('\r', '')]


DC_HQPC_current_bias_resistance = 10e6  # Resistor inline with DC bias for HQPC