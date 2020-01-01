import pickle
import inspect
from unittest.mock import patch
from src import temp
import pandas as pd
import numpy as np

def replacementfn(inp):
    print(f'Im the replacement and I got the input: {inp}')
    return 'fake answer'


df = pd.DataFrame([[1,2,3], [4, 5, 6]], columns = ['one', 'two', 'three'])



def add_col_label(df, new_col, on_cols, level=1):
    dfinternal = df[:]
    if level == 0:
        raise ValueError("Using level 0 will overwrite main column titles")
    if type(on_cols) != list:
        on_cols = [on_cols]

    if isinstance(dfinternal.columns, pd.Index) and not isinstance(dfinternal.columns, pd.MultiIndex):  # if only 1D index, must be asking for new column level
        newcols = _new_level_emptycols(dfinternal)
    elif len(dfinternal.columns.levels)-1 < level:  # if asking for new level
        newcols = _new_level_emptycols(dfinternal, level=level)
    else:  # column labels already exist
        if type(on_cols[0]) == tuple:  #if fully addressing with tuples
            newcols = _existing_level_cols(dfinternal, level, address='full')
        else:  # Address by top col only
            newcols = _existing_level_cols(dfinternal, level, address='top')
    for col in on_cols:  # Set new values of columns
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

def _new_level_emptycols(df, level=1):
    if level == 1:
        return dict(zip(df.columns, np.repeat('', df.shape[1])))
    else:
        return dict(zip(df.columns.levels[0], np.repeat('', df.shape[1])))

def _existing_level_cols(df, level=1, address = 'top'):
    newcols = [x[level] for x in list(df.columns)]
    if address == 'top':
        newcols = dict(zip([x[0] for x in df.columns], newcols))
    elif address == 'full':
        newcols = dict(zip(df.columns.levels, newcols))
    return newcols

if __name__ == '__main__':
    print(df)
    df2 = add_col_label(df, '2nd', ['two', 'three'])
    df3 = add_col_label(df2, '3rd', ['one', 'three'], level=1)
    df4 = add_col_label(df2, '3rd', ['one', 'three'], level=2)
    
    

