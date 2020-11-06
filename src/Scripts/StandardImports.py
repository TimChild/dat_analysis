"""Has most imports for normal plotting scripts"""
#  Packages from my code that I use often
from src import CoreUtil as CU
from src.DatObject.Make_Dat import DatHandler

#  Common packages I use
import logging

#  The Experiment's I'm currently working with. Makes it easier to get to Config/ESI/Fixes
import src.DataStandardize.ExpSpecific.Aug20 as Aug20
#import src.DataStandardize.ExpSpecific.Jun20 as Jun20
# import src.DataStandardize.ExpSpecific.Jan20 as Jan20

# Most commonly used functions and classes
get_dat = DatHandler.get_dat
get_dats = DatHandler.get_dats
AugESI = Aug20.AugESI
# JunESI = Jun20.JunESI
# JanESI = Jan20.JanESI
logger = logging.getLogger('MAIN')

# # Set logging defaults
root_logger = logging.getLogger()
if not root_logger.handlers:
    CU.set_default_logging()

