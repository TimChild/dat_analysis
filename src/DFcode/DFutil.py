import pandas as pd
from functools import wraps
from src import config as cfg
from src.CoreUtil import verbose_message
import pickle
import os



def getexceldf(path, comparisondf=None, dtypes:dict=None) -> pd.DataFrame:
    """Returns excel df at given path, or will ask for user input comparison df provided"""
    if os.path.isfile(path):
        exceldf = pd.read_excel(path, index_col=0, header=0, dtype=dtypes)
        assert exceldf.index.name == 'datnumplus'
        if comparisondf is not None:
            df = compare_pickle_excel(comparisondf, exceldf,
                                              f'SetupDF')  # Returns either pickle or excel df depending on input
        else:
            df = exceldf
    elif comparisondf is not None:
        df = comparisondf
    else:
        raise ValueError(f'No dataframe to return, none given as "dfcompare" param and none at [{path}]')
    return df


def open_xlsx(filepath):
    def _is_excel(filepath):
        if filepath[-4:] == 'xlsx':
            return True
        else:
            raise TypeError(f'Filepath points to a non-excel file at "{filepath}"')

    if _is_excel(filepath):
        if os.path.isfile(filepath):
            os.startfile(filepath)
        else:
            raise FileNotFoundError(f'No file found at "{filepath}"')
    return None


def change_in_excel(path) -> pd.DataFrame:
    """Lets user change df in excel. Does not save changes by default!!!"""
    open_xlsx(path)
    input(f'After finished editing and changes are saved press any key to continue')
    df = getexceldf(path)
    # region Verbose SetupDF change_in_excel
    if cfg.verbose is True:
        verbose_message('DF loaded from excel. Not saved to pickle by default!!!')
    # endregion
    return df


def protect_data_from_reindex(func):
    if getattr(func, '_decorated', False):  # Prevent double wrapping of function
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        df = args[0]  # type: pd.DataFrame
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
        inst.filepathpkl = path
        if not isinstance(inst, cls):  # Check if loaded version is actually a setupPD
            raise TypeError(f'File saved at {path} is not of the type {cls}')
        return inst
    else:
        return None


def temp_reset_index(func):
    """Decorator that temporarily resets index then returns it to what it was before. Requires either df as first
     argument or an object a df at obj.df as first argument. Mostly to be used on SetupDF or DatDF methods"""
    _flag = None

    def _getdfindexnames(*args):
        nonlocal _flag
        if isinstance(args[0], pd.DataFrame):
            df = args[0]
            _flag = 0
        elif isinstance(args[0].df, pd.DataFrame):
            df = args[0].df
            _flag = 1
        else:
            raise ValueError('temp_reset_index requires df or object with .df as first arg')
        indexnames = df.index.names
        return indexnames

    def _setdfindex(*args, indexnames: list = (None)):
        nonlocal _flag
        if indexnames[0] is not None:
            if _flag == 0:
                _resetdfindex(*args)
                args[0].set_index(indexnames, inplace=True)
            elif _flag == 1:
                _resetdfindex(*args)
                args[0].df.set_index(indexnames, inplace=True)
        return None

    def _resetdfindex(*args):
        nonlocal _flag
        if _flag == 0:
            if args[0].index.names[0] is not None:
                args[0].reset_index(inplace=True)
        elif _flag == 1:
            if args[0].df.index.names[0] is not None:
                args[0].df.reset_index(inplace=True)
        return None

    @wraps(func)
    def wrapper(*args, **kwargs):
        indexnames = _getdfindexnames(*args)
        _resetdfindex(*args)
        ret = func(*args, **kwargs)
        _setdfindex(*args, indexnames=indexnames)
        return ret

    return wrapper
