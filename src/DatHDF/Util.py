import src.Configs.Main_Config as cfg
import os
import h5py


def get_dat_hdf_path(dat_id, dir=None):
    dir = dir if dir else cfg.hdfdir
    file_path = os.path.join(dir, dat_id+'.h5')
    if not os.path.exists(file_path):  # make empty file then return path
        dir, _ = os.path.split(file_path)
        os.makedirs(dir, exist_ok=True)  # Ensure directory exists
        f = h5py.File(file_path, 'w')  # Init a HDF file
        f.close()
    return file_path