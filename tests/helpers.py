import src.config as cfg
import functools
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
    def set_test_dirs():  # So tests can always point to same Dat files
        # ddir = os.path.join('src/Data/Experiment_Data')
        cfg.ddir = os.path.join('tests/fixtures/dats')
        pickledata = os.path.join('tests/fixtures/pickles')
        plotdir = os.path.join('tests/fixtures/plots')
        dfdir = os.path.join('tests/fixtures/DataFrames')

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
