from unittest import TestCase
from tests import helpers as th
from src import Sep19ExperimentSpecific as ES
import pandas as pd
from src.DFcode.DFutil import protect_data_from_reindex
import src.DFcode.SetupDF as SDF
import src.DFcode.DatDF as DDF
th.setverbose(False, 20)
th.Dirs.set_test_dirs()
pd.DataFrame.set_index = protect_data_from_reindex(pd.DataFrame.set_index)


class TestAddingDats(TestCase):
    def Test_setupdf(self):
        sdf = SDF.SetupDF()
        self.assertequal('D:\\OneDrive\\UBC LAB\\GitHub\\Python\\PyDatAnalysis/tests/fixtures/unit\\DataFrames/setup'
                         '/setup.pkl', sdf.filepathpkl)
        self.assertEqual('D:\\OneDrive\\UBC LAB\\GitHub\\Python\\PyDatAnalysis/tests/fixtures/unit\\DataFrames/setup'
                         '/setup.xlsx', sdf.filepathexcel)

    def Test_load_dat_from_df(self):
        sdf = SDF.SetupDF()
        dat = ES.make_dat_standard(2713, dfname='BasicWork')


if __name__ == '__main__':
    sdf = SDF.SetupDF()
    # sdf.change_in_excel()
    sdf.save()
    ddf = DDF.DatDF(dfname='BasicWork')
    ddf.save()
    datnums = list(range(2713, 2720+1))  # 100 line repeats along 0->1 at 200mV/s
    dat = ES.make_dat_standard(2713, dfname='BasicWork', dattypes=['isense'])

