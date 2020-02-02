import numpy as np
import src.DatCode.DatAttribute as DA
import h5py


class Data(DA.DatAttribute):
    """Stores all raw data of Dat"""

    def __init__(self, infodict=None, hdfpath=None):
        """Creates Data Attribue for Dat. Can specify path to hdf if desired"""
        self.data_keys = None
        if infodict is not None:
            self.x_array = infodict['Logs'].get('x_array')
            self.y_array = infodict['Logs'].get('y_array')
            self.i_sense = infodict.get('i_sense')
            self.entx = infodict.get('entx')
            self.enty = infodict.get('enty')
            self.conductance = infodict.get('conductance')
            self.current = infodict.get('current')
            if hdfpath is None:
                hdfpath = infodict.get('hdfpath')
        self.hdfpath = hdfpath
        if self.hdfpath is not None:
            hdf = h5py.File(self.hdfpath, 'r')
            keylist = hdf.keys()
            data_keys = []
            for key in keylist:
                if isinstance(hdf[key], h5py.Dataset):  # Make sure it's a dataset not metadata
                    data_keys.append(key)
            self.data_keys = data_keys
            
    def __getattr__(self, item):
        """Overrides behaviour when attribute is not found for Data"""
        if item in self.data_keys:
            hdf = h5py.File(self.hdfpath, 'r')
            return hdf[item]
        else:
            print(f'Dataset "{item}" does not exist for this Dat')
            return None

    def __getstate__(self):
        """Required for pickling because of __getattr__ override"""
        return self.__dict__

    def __setstate__(self, state):
        """Required for unpickling because of __getattr__ override"""
        self.__dict__.update(state)


    def get_names(self):
        """Returns list of data names that is not None"""
        datanames = [attrname for attrname in self.__dict__.keys() if getattr(self, attrname, None) is not None]
        datanames += self.data_keys
        datanames.remove('data_keys')
        datanames.remove('hdfpath')
        return datanames

