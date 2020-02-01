"""All instrument information in here. I.e. SRS settings, Magnet settings etc"""
from typing import Dict
from src.DatCode.DatAttribute import DatAttribute
import src.DatCode.DatAttribute as DA


class Instruments(DatAttribute):
    """Dat attribute which contains all instrument settings as attributes of itself. Each instrument should be a
    namedtuple of it's info"""

    def __init__(self, infodict: Dict):
        self.srs1 = DA.get_instr_vals('srs', 1, infodict)
        self.srs2 = DA.get_instr_vals('srs', 2, infodict)
        self.srs3 = DA.get_instr_vals('srs', 3, infodict)
        self.srs4 = DA.get_instr_vals('srs', 4, infodict)

        self.magx = DA.get_instr_vals('mag', 'x', infodict)
        self.magy = DA.get_instr_vals('mag', 'y', infodict)
        self.magz = DA.get_instr_vals('mag', 'z', infodict)





