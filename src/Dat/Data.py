import numpy as np
import src.Dat.DatAttribute as DA


class Data(DA.DatAttribute):
    """Stores all raw data of Dat"""
    def __init__(self, infodict):
        self.x_array = infodict['x_array']
        self.y_array = infodict['y_array']
        self.i_sense = infodict['i_sense']
        self.entx = infodict['entx']
        self.enty = infodict['enty']

    def __setattr__(self, key, value):
        """Sets attributes if possible, but allows for key error in infodict"""
        try:
            super().__setattr__(key, value)
        except KeyError:
            return None

    def get_names(self):
        """Returns list of data names that is not None"""
        datanames = [attrname for attrname in self.__dict__.keys() if getattr(self, attrname, None) is not None]
        return datanames
