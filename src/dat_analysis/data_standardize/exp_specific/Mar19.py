"""
2022 -- Way too out of date...
"""


import os

path_replace = None

instruments = {'srs': 'srs830', 'dmm': 'hp34401a', 'dac': 'babydac', 'magnet':'ls625', 'fridge':'ls370'}
instrument_num = {'srs': 4, 'dmm': 1, 'dac': 16, 'magnet': 3}

dat_types_list = ['none', 'i_sense', 'entropy', 'transition', 'dcbias']

### Name of directory with all Data (e.g. dats, dataframes, pickles etc). Hopefully this will allow moving out of project without
#  losing access to everything
dir_name = ''  # Name of folder inside main data folder specified by dat_analysis.config.main_data_path

wavenames = ['x_array', 'y_array', 'i_sense', 'i_sense2d', 'g1x2d', 'g1y2d', 'g2x2d', 'g2y2d']  # Common names of waves used
raw_wavenames = ['v5dc', 'v5dc2d']  # Pre calc waves that are effectively duplicated above

i_sense_keys = ['i_sense', 'i_sense2d']  # Will prioritize last values
entropy_x_keys = ['g2x', 'g2x2d']
entropy_y_keys = ['g2y', 'g2y2d']

# li_theta_x_keys = ['g3x']
# li_theta_y_keys = ['g3y']

json_subs = []

# json_subs = [('"comment": "{"gpib_address":4, "units":"VOLT", "range":.1.000000E.0., "resolution":...000000E-0.}"',
#                 '"comment": "replaced to fix json"'),
#             (":\+", ':'),
#             ('\r', '')]


DC_HQPC_current_bias_resistance = 10e6  # Resistor inline with DC bias for HQPC



#### Any experiment specific fixing functions can go here... e.g.



# def add_magy(dats):
#     i = 0
#     from dat_analysis.DFcode.DatDF import DatDF
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
#     import dat_analysis.Configs.Main_Config as cfg
#     import dat_analysis.Core as C
#     import dat_analysis.DFcode.DatDF as DF
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