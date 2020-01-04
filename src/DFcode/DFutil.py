import pandas as pd
from functools import wraps
from src import config as cfg
from src.CoreUtil import verbose_message
import pickle
import os


def protect_data_from_reindex(func):
    if getattr(func, '_decorated', False):  # Prevent double wrapping of function
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        df = args[0] # type: pd.DataFrame
        assert isinstance(df, pd.DataFrame)
        if df.index.names[0] is not None:
            df.reset_index(inplace=True)
            print(f'WARNING: You were just saved from deleting the data [{df.index.names}] while reassigning index.'
                  f'\nYou might need to keep track of your index more closely')
        ret = func(*args, **kwargs)
        return ret
    wrapper._decorated = True
    return wrapper

def compare_pickle_excel(dfpickle, dfexcel, name='...[NAME]...') -> pd.DataFrame:
    df = dfpickle
    if _compare_to_df(dfpickle, dfexcel) is False:
        inp = input(f'DFs for {name} have a different pickle and excel version of DF '
                    f'do you want to use excel version?')
        if inp.lower() in {'y', 'yes'}:  # Replace pickledf with exceldf
            df = dfexcel
    return df


def _compare_to_df(dfone, dftwo):
    """Returns true if equal, False if not.. Takes advantage of less precise comparison
    of assert_frame_equal in pandas"""
    try:
        pd.testing.assert_frame_equal(dfone, dftwo)
        return True
    except AssertionError as e:
        # region Verbose DatDF compare_to_df
        if cfg.verbose is True:
            verbose_message(f'Verbose[][_compare_to_df] - Difference between dataframes is [{e}]')
        # endregion
        return False


def load_from_pickle(path, cls):
    """Returns instance of class at path if it exists, otherwise returns None"""
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            inst = pickle.load(f)  # inspect.stack()[1][3] == __new__ required to stop loop here
        inst.loaded = True
        if not isinstance(inst, cls):  # Check if loaded version is actually a setupPD
            raise TypeError(f'File saved at {path} is not of the type {cls}')
        return inst
    else:
        return None