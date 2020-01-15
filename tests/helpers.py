import src.config as cfg
import functools
import inspect
import os



class Dirs:
    """For changing directories for testing purposes"""
    ddir = cfg.ddir  # Save values already in config so they can be restored
    pickledata = cfg.pickledata
    plotdir = cfg.plotdir
    dfdir = cfg.dfdir

    def __init__(self):
        pass

    @staticmethod
    def set_test_dirs(which='unit'):  # So tests can always point to same Dat files
        if which not in ['unit', 'integration']:
            raise ValueError (f'Test dirs need to be set for "unit" or "integration"')
        abspath = os.path.abspath('.').split('PyDatAnalysis')[0]
        abspath = os.path.join(abspath, f'PyDatAnalysis/tests/fixtures/{which}')
        cfg.ddir = os.path.join(abspath, 'dats')
        cfg.pickledata = os.path.join(abspath, 'pickles')
        cfg.plotdir = os.path.join(abspath, 'plots')
        cfg.dfdir = os.path.join(abspath, 'DataFrames')
        cfg.dfsetupdirpkl = os.path.join(abspath, 'DataFrames/setup/setup.pkl')
        cfg.dfsetupdirexcel = os.path.join(abspath, 'DataFrames/setup/setup.xlsx')

    @staticmethod
    def reset_dirs():
        print(f'Setting cfg.dir to {Dirs.ddir}')
        cfg.ddir = Dirs.ddir
        cfg.pickledata = Dirs.pickledata
        cfg.plotdir = Dirs.plotdir
        cfg.dfdir = Dirs.dfdir


def change_to_test_dir(func):
    """Temporarily changes all directories to test directories"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        Dirs.set_test_dirs()
        value = func(*args, **kwargs)
        Dirs.reset_dirs()
        return value
    return wrapper


def stackinspecter():
    stack = inspect.stack()
    for i, frame in enumerate(stack):
        for j, val in enumerate(frame):
            print(f'[{i}][{j}] = {val},', end='\t')
        print('')


def simple_mock_input(return_list):
    if type(return_list) != list:
        return_list = [return_list]
    i=0
    def _fake_input(inp):
        nonlocal i
        ret = return_list[i]
        i+=1
        print(inp)
        print(ret)
        return ret
    return _fake_input


# def change_to_mock_input(mod, inputs: list, changefunc = 'input'):
#     def change_to_mock_decorator(fit_function):
#         @functools.wraps(fit_function)
#         def wrapper(*args, **kwargs):
#             # region Verbose change_to_mock_input wrapper
#             if cfg.verbose is True:
#                 print('Using change_to_mock_input wrapper')
#             # endregion
#
#             normalfunc = mod.__builtins__[changefunc]
#             print(type(normalfunc))
#             if not isinstance(normalfunc, type(open)):  # avoid wrapping multiple times (open is just an arbitrary builtin)
#                 return fit_function(*args, **kwargs)
#             mockinputs = MagicMock()
#             mockinputs.side_effect = inputs
#             mod.__builtins__[changefunc] = inputwrapper(mod.__builtins__[changefunc], mockinputs)
#             ret = fit_function(*args, **kwargs)
#             mod.__builtins__[changefunc] = normalfunc  # Reset behaviour
#             return ret
#
#         return wrapper
#     return change_to_mock_decorator


def inputwrapper(func, mockinputs):
    @functools.wraps(func)
    def wrapper(*args):
        print(args[0])
        ret = mockinputs()
        print(ret)
        return ret
    return wrapper


def setverbose(truth:bool, level:int = 20):
    cfg.verbose = truth
    cfg.verboselevel = level