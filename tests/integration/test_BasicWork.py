from unittest import TestCase
from tests import helpers as th
from src import Sep19ExperimentSpecific as ES
import pandas as pd
from src.DFcode.DFutil import protect_data_from_reindex
th.setverbose(False, 20)
th.Dirs.set_test_dirs()
pd.DataFrame.set_index = protect_data_from_reindex(pd.DataFrame.set_index)


# class TestAddingDats(TestCase):
#     def
