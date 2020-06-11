import os

path_replace = None  # ('work\\Fridge Measurements with PyDatAnalysis', 'work\\Fridge_Measurements_and_Devices\\Fridge Measurements with PyDatAnalysis')

instruments = {'srs': 'srs830', 'dmm': 'hp34401a', 'dac': 'babydac', 'fastdac': 'fastdac', 'magnet':'ls625', 'fridge':'ls370'}
instrument_num = {'srs': 3, 'dmm': 1, 'dac': 16, 'fastdac': 8, 'magnet': 3}

dat_types_list = ['none', 'i_sense', 'entropy', 'transition', 'pinch', 'dot tuning', 'dcbias']

### Path to all Data (e.g. dats, dataframes, pickles etc). Hopefully this will allow moving out of project without
#  losing access to everything
abspath = os.path.abspath('..').split('PyDatAnalysis')[0]
dir_name = 'Jun20'  # Name of folder inside main data folder specified by src.config.main_data_path

wavenames = ['x_array', 'y_array', 'i_sense', 'entx', 'enty']
raw_wavenames = [f'ADC{i:d}' for i in range(4)] + [f'ADC{i:d}_2d' for i in range(4)] + ['g1x', 'g1y', 'g2x', 'g2y']

x_array_keys = ['x_array']
y_array_keys = ['y_array']
i_sense_keys = ['i_sense', 'cscurrent', 'cscurrent_2d']
entx_keys = ['entx', 'entropy_x_2d', 'entropy_x']
enty_keys = ['enty', 'entropy_y_2d', 'entropy_y']
# li_theta_x_keys = ['g3x']
# li_theta_y_keys = ['g3y']

# json_subs = [('"comment": "{"gpib_address":4, "units":"VOLT", "range":.1.000000E.0., "resolution":...000000E-0.}"',
#                 '"comment": "replaced to fix json"'),
#              (":\+", ':'),
#              ('\r', '')]
json_subs = []

DC_HQPC_current_bias_resistance = 10e6  # Resistor inline with DC bias for HQPC




# def add_magy(dats):
#     i = 0
#     from src.DFcode.DatDF import DatDF
#     datdf = DatDF()
#     for dat in dats:
#         if dat.Logs.magy is None:
#             for string in dat.Logs.comments.split(','):
#                 if string[:4] == 'magy':
#                     print(f'Updating Dat{dat.datnum}: Magy = {string[5:-2]}mT')
#                     dat.Logs.magy = float(string[5:-2])
#                     datdf.update_dat(dat)
#                     i+=1
#         if dat.Logs.magy is None:
#             print(f'dat{dat.datnum} has no magy in comments')
#     if i != 0:
#         print(f'saving {i} updates to df')
#         datdf.save()


# def add_mag_to_logs(dats):
#     import h5py
#     import src.Configs.Main_Config as cfg
#     import src.Core as C
#     import src.DFcode.DatDF as DF
#     datdf = DF.DatDF()
#     i =0
#     for dat in dats:
#         if dat.Logs.magy is None:
#             hdf = h5py.File(dat.Data.hdfpath, 'r')
#             sweeplogs = hdf['metadata'].attrs['sweep_logs']
#             sweeplogs = C.metadata_to_JSON(sweeplogs)
#             mags = {'mag' + id: C._mag_from_json(sweeplogs, id, mag_type=instruments['magnet']) for id in ['x', 'y', 'z']}
#             dat.Logs.add_mags(mags)
#             dat.Instruments.add_mags(mags)
#             dat.Instruments.field_y = dat.Instruments.magy.field
#             cfg.yes_to_all = True
#             datdf.update_dat(dat)
#             cfg.yes_to_all = False
#             i+=1
#     if i != 0:
#         print(f'saving {i} updates to df')
#         datdf.save()