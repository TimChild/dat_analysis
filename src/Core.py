"""Core of PyDatAnalysis. This should remain unchanged between experiments in general, or be backwards compatible"""

import json
import os
import pickle
import re
import src.config as cfg

################# Settings for Debugging #####################
from src.Dat.Dat import Dat
from src.DFcode.DatDF import DatDF, _dat_exists_in_df

cfg.verbose = True
cfg.verboselevel = 19  # Max depth of stack to have verbose prints from
timer = False


###############################################################


################# Other configurables #########################
#
# if platform.system() == "Darwin":  # Darwin is Mac... So I can work with Owen
#     os.chdir("/Users/owensheekey/Documents/Research/One_CK_analysis/OneCK-Analysis")
# else:
#     os.chdir('D:/OneDrive/UBC LAB/GitHub/Python/PyDatAnalysis/')

##Just reminder for what exists
# cfg.ddir
# cfg.pickledata
# cfg.plotdir
# cfg.dfdir

###############################################################

################# Sweeplog fixes ##############################


def metadata_to_JSON(data: str) -> dict:  # TODO, FIXME: Move json edits into experiment specific
    data = re.sub(', "range":, "resolution":', "", data)
    data = re.sub(":\+", ':', data)
    try:
        jsondata = json.loads(data)
    except json.JSONDecodeError:
        data = re.sub('"CH0name".*"com_port"', '"com_port"', data)
        jsondata = json.loads(data)
    return jsondata


###############################################################

################# Constants and Characters ####################

# Temporary comment: Moved to Characters.py
###############################################################


################# Beginning of Classes ########################


def datfactory(datnum, datname, dfname, dfoption, infodict=None):
    datdf = DatDF(dfname=dfname)  # Load DF
    datcreator = _creator(dfoption)  # get creator of Dat instance based on df option
    datinst = datcreator(datnum, datname, datdf, infodict)  # get an instance of dat using the creator
    return datinst  # Return that to caller


def _creator(dfoption):
    if dfoption in ['load', 'load_pickle']:
        return _load_pickle
    if dfoption in ['load_df']:
        return _load_df
    elif dfoption == 'sync':
        return _sync
    elif dfoption == 'overwrite':
        return _overwrite
    else:
        raise ValueError("dfoption must be one of: load, sync, overwrite, load_df")


def _load_pickle(datnum: int, datname, datdf, infodict=None):
    _dat_exists_in_df(datnum, datname, datdf)
    datpicklepath = datdf.get_path(datnum, datname=datname)

    if os.path.isfile(datpicklepath) is False:
        inp = input(f'Pickle for dat{datnum}[{datname}] doesn\'t exist in "{datpicklepath}", would you like to load using DF[{datdf.name}]?')
        if inp in ['y', 'yes']:
            return _load_df(datnum, datname, datdf, infodict)
        else:
            raise FileNotFoundError(f'Pickle file for dat{datnum}[{datname}] doesn\'t exist')
    with open(datpicklepath) as f:  # TODO: Check file exists
        inst = pickle.load(f)
    return inst


def _load_df(datnum: int, datname:str, datdf, *args):
    # TODO: make infodict from datDF then run overwrite with same info to recreate Datpickle
    _dat_exists_in_df(datnum, datname, datdf)
    infodict = datdf.infodict(datnum, datname)
    inst = _overwrite(datnum, datname, datdf, infodict)
    return inst


def _sync(datnum, datname, datdf, infodict):
    if (datnum, datname) in datdf.df.index:
        inp = input(f'Dat{datnum}[{datname}] already exists, do you want to \'load\' or \'overwrite\'')
        if inp == 'load':
            inst = _load_pickle(datnum, datname, infodict, )
        elif inp == 'overwrite':
            inst = _overwrite(datnum, datname, infodict)
        else:
            raise ValueError('Must choose either \'load\' or \'overwrite\'')
    else:
        inst = _overwrite(datnum, datname, datdf, infodict)
    return inst


def _overwrite(datnum, datname, datdf, infodict):
    inst = Dat(datnum, datname, infodict, dfname=datdf.name)
    return inst


#


################# End of classes ################################

################# Functions #####################################


def coretest(nums):
    """For testing unit testing"""
    result = 0
    for num in nums:
        result += num
    return result


