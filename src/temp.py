import pickle
import inspect
from unittest.mock import patch
from src import temp
import pandas as pd
import numpy as np
import src.DFcode.DFutil as DU









iterables = [['bar', 'baz', 'foo', 'qux'], ['one', 'two']]
index = pd.MultiIndex.from_product(iterables, names=['first', 'second'])


df = pd.DataFrame(np.random.randn(3, 8), index=['A', 'B', 'C'], columns=index)
df = DU.add_col_label(df, 'new', ('qux', 'one'), level=2)

