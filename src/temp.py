import pickle
import inspect
from unittest.mock import patch
from src import temp
import pandas as pd
import numpy as np


def add_col_label(df, new_col, on_cols, level=1):
    def _new_level_emptycols(df, level=1, address='top'):
        if level == 1:
            return dict(zip(df.columns, np.repeat('', df.shape[1])))
        else:
            if address == 'full':
                return dict(zip([x for x in df.columns], np.repeat('', df.shape[1])))
            elif address == 'top':
                return dict(zip([x[0] for x in df.columns], np.repeat('', df.shape[1])))
            else:
                raise ValueError(f'Address "{address}" is not valid, choose "top" or "full"')

    def _existing_level_cols(df, level=1, address='top'):
        newcols = [x[level] for x in list(df.columns)]
        if address == 'top':
            newcols = dict(zip([x[0] for x in df.columns], newcols))
        elif address == 'full':
            newcols = dict(zip(df.columns.levels, newcols))
        return newcols

    def _newcols_generator(dfinternal, level):
        if isinstance(dfinternal.columns, pd.Index) and not isinstance(dfinternal.columns,
                                                                       pd.MultiIndex):  # if only 1D index, must be asking for new column level
            newcolsfn = _new_level_emptycols
        elif len(dfinternal.columns.levels) - 1 < level:  # if asking for new level
            newcolsfn = _new_level_emptycols
        else:  # column labels already exist
            newcolsfn = _existing_level_cols
        return newcolsfn

    dfinternal = df[:]  # shallow copy to prevent changing later df's
    if level == 0:
        raise ValueError("Using level 0 will overwrite main column titles")
    if type(on_cols) != list:
        on_cols = [on_cols]
    if type(on_cols[0]) == tuple:  # if fully addressing with tuples
        address = 'full'
    else:
        address = 'top'

    newcolsfn = _newcols_generator(df, level)  # Either gets _new... or _existing... colnames
    newcols = newcolsfn(dfinternal, level, address=address)

    for col in on_cols:  # Set new values of columns
        if col not in newcols.keys():
            raise KeyError(f'Column ({col}) does not exist in df')
        newcols[col] = new_col
    if isinstance(dfinternal.columns, pd.Index) and not isinstance(dfinternal.columns, pd.MultiIndex):
        colarray = [list(dfinternal.columns)]
    else:
        colarray = []
        for i in [x for x in range(len(dfinternal.columns.levels)) if x != level]:
            colarray.append([x[i] for x in dfinternal.columns])
    colarray.append(list(newcols.values()))
    dfinternal.columns = pd.MultiIndex.from_arrays(colarray)
    return dfinternal






iterables = [['bar', 'baz', 'foo', 'qux'], ['one', 'two']]
index = pd.MultiIndex.from_product(iterables, names=['first', 'second'])


df = pd.DataFrame(np.random.randn(3, 8), index=['A', 'B', 'C'], columns=index)
df = add_col_label(df, 'new', ('qux', 'one'), level=2)

