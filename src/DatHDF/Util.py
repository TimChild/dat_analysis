import src.Configs.Main_Config as cfg
import os
import h5py
import numpy as np


def get_dat_hdf_path(dat_id, dir=None, overwrite=False):
    dir = dir if dir else cfg.hdfdir
    file_path = os.path.join(dir, dat_id+'.h5')
    if os.path.exists(file_path) and overwrite is True:
        os.remove(file_path)
    if not os.path.exists(file_path):  # make empty file then return path
        dir, _ = os.path.split(file_path)
        os.makedirs(dir, exist_ok=True)  # Ensure directory exists
        f = h5py.File(file_path, 'w')  # Init a HDF file
        f.close()
    return file_path

#
# def save_dict(in_group: h5py.Group, name: str, dictionary: dict):
#     """Save a dictionary called <name> within <in-group>"""
#     dict_group = in_group.create_group(name)
#     dict_group.attrs['is_dictionary'] = True
#     for k, v in dictionary.items():
#         dict_group.create_dataset(k, data=v)
#
#
# def load_dict(dict_group: h5py.Group):
#     """Loads a dictionary from a group which has an 'is_dictionary' attr """
#     assert dict_group.attrs['is_dictionary'] is True
#     d = {}
#     for key in dict_group.keys():
#         if isinstance(dict_group[key], h5py.Dataset):
#             d[key] = dict_group[key]
#     return d
