"""
For making ExpConfig group in DatHDF

Idea of ExpConfig group is to store any additional information which is not in ExpDat directly in a way that allows
DatAttrs to initialize properly.

e.g. Dat.Data should store 'i_sense' data in nA, but the data recorded might have been 10x smaller and called
'cscurrent', so there needs to be something somewhere which says that the 'cscurrent' is the 'i_sense' data,
but it needs to be multiplied by 10 first

Also can store info in here like voltage divider set ups which otherwise aren't necessarily stored in Exp data

ExpConfigBase should be subclassed for each new cooldown in general to override whatever needs to be changed for that
exp directly. For one off changes, modifying the DatHDFs directly is also a good options, just be careful not to
overwrite it as the info to reinitialize properly won't be there """
from __future__ import annotations
import abc
from src.DatObject.Attributes.DatAttribute import DatAttribute
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF

from src import CoreUtil as CU


class ExpConfigBase(abc.ABC):
    """
    Base Config class to outline what info needs to be in any exp specific config

    Should be things which are Experiment specific but not stored in the HDFs directly
    # TODO: Make something where this info can be stored/read from a JSON file (i.e. so easier to modify with out going to code)
    """

    def __init__(self, datnum=None):
        self.datnum = datnum

    @abc.abstractmethod
    def get_sweeplogs_json_subs(self):
        """Something that returns a list of re match/repl strings to fix sweeplogs JSON for a given datnum.
        i.e. if some of the sweeplogs saved were not valid JSON's then JSON parse will not work until things are fixed

        Form: [(match, repl), (match, repl),..]
        """
        return [('FastDAC 1', 'FastDAC')]

    # @abc.abstractmethod
    # def get_dattypes_list(self) -> set:
    #     """Something that returns a list of dattypes that exist in experiment"""
    #     return {'none', 'entropy', 'transition', 'square entropy'}

    @abc.abstractmethod
    def get_exp_names_dict(self) -> dict:
        """Override to return a dictionary of experiment wavenames for each standard name
        standard names are: i_sense, entx, enty, x_array, y_array"""
        d = dict(x_array=['x_array'], y_array=['y_array'],
                 i_sense=['cscurrent', 'cscurrent_2d'])
        return d


class ExpConfigGroupDatAttribute(DatAttribute):
    """
    Given a HDFContainer and
    """
    version = '1.0.0'
    group_name = 'Experiment Config'
    description = 'Information about the experiment which is not stored in the dat file directly. ' \
                  'e.g. potential dividers used, current amplifier settings, or anything else you want. ' \
                  'Can also include things that are required to fix the data recorded after the fact. ' \
                  'e.g. parts of sweeplogs which need to be replaced to make them valid JSONs, rows of data to avoid...'

    def _initialize_minimum(self):
        self.initialized = True

    def __init__(self, dat: DatHDF, exp_config: ExpConfigBase = None):
        super().__init__(dat)
        self.exp_config = exp_config



