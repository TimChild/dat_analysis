from unittest import TestCase
from src import Core
from src import ExampleExperimentSpecific as EES
import tests.helpers as th
import os
from unittest.mock import patch
from tests.unit.test_ExperimentSpecific import Test_make_dat as make
import src.config as cfg
from tests.helpers import change_to_mock_input
th.Dirs.set_test_dirs()
th.setverbose(False, level=25)


class TestDatDF(TestCase):

    @staticmethod
    def makefilledDF(name):
        """Makes DF with some information. Maybe want to put this somewhere else if it gets bigger"""
        datDF = Core.DatDF(dfname=name)
        inputs = ['y', 'y', 'y', 'y', 'y', 'y']
        EES.make_dat_standard = change_to_mock_input(EES, inputs)(EES.make_dat_standard)
        dat = EES.make_dat_standard(2700, dfname='test_add')  # TODO: add inputs
        datDF.save()

    @staticmethod
    def getfilledDF(name):
        if not os.path.isfile(f'../fixtures/DataFrames/{name}.pkl'):
            TestDatDF.makefilledDF(name)
        return Core.DatDF(dfname=name)

    @staticmethod
    def killDF(name):
        for savetype in ['pkl', 'xlsx']:
            if os.path.isfile(f'../fixtures/DataFrames/{name}.{savetype}'):
                os.remove(f'../fixtures/DataFrames/{name}.{savetype}')

    @staticmethod
    def getcleanDF(name):
        TestDatDF.killDF(name)
        return Core.DatDF(dfname=name)

    def test_newDF(self):  # Removes then creates new empty DF
        """Just tests that creating a new dat makes a pkl and xlsx file"""
        datDF = TestDatDF.getcleanDF('new')
        self.assertEqual(datDF.loaded, False)
        self.assertTrue(os.path.isfile('../fixtures/DataFrames/new.pkl'))
        self.assertTrue(os.path.isfile('../fixtures/DataFrames/new.xlsx'))

    def test_add_dat(self):
        """Adding row of dat values to DF (this will need modifying when more info is added to dats)"""
        datDF = TestDatDF.getcleanDF('test_add')
        inputs = ['y', 'y', 'y', 'y', 'y', 'y']
        EES.make_dat_standard = change_to_mock_input(EES, inputs)(EES.make_dat_standard)
        dat = EES.make_dat_standard(2700, dfname='test_add')
        inputs = ['y', 'y', 'y', 'y', 'y']  # Adding new columns to clean DF
        EES.Dat.savetodf = change_to_mock_input(EES, inputs)(EES.Dat.savetodf)
        dat.savetodf(dfname='test_add')

    def test_add_dat_attr(self):
        """Make sure correct things are rejected/allowed to be entered in DF"""
        datDF = TestDatDF.getcleanDF('test_add_attr')
        for col, val in zip(['stringcol', 'intcol', 'floatcol'], ['string', '5', '5.4']):
            with patch('builtins.input', return_value='y'):  # Just enters yes when asked to add new columns
                datDF.add_dat_attr(datnum=1, attrname=col, attrvalue=val)
            self.assertEqual(datDF.df.at[1, col], val)
        import numpy as np
        for col, val in zip(['dfname', 'array', 'dict'], ['name', np.array([1,1]), {'a': 1}]):
            datDF.add_dat_attr(datnum=1, attrname=col, attrvalue=val)
            with self.assertRaises(KeyError):
                print(datDF.df.at[1, col])
        datDF.add_dat_attr(datnum=1, attrname='datnum', attrvalue=10)
        self.assertTrue(np.isnan(datDF.df.at[1, 'datnum']))

    def test_save_load(self):
        """Tests that DF saves correctly, Tests if loading without killing open DF opens same instance, 
        Tests if loading from file give same df"""
        
        datDF = TestDatDF.getfilledDF('test_save_load')
        TestDatDF.killDF('save')
        datDF.save('save')
        for ext in ['pkl', 'xlsx']:
            self.assertTrue(os.path.isfile(f'../fixtures/DataFrames/save.{ext}'))

        datDF2 = EES.DatDF(dfname='save')
        self.assertEqual(datDF, datDF2) # Should get same instance and not load from file

        EES.DatDF.killinstances()
        datDF3 = EES.DatDF(dfname='save')
        self.assertTrue(datDF.df.equals(datDF3.df))  # should have to load from file
        

    def test_sync_dat(self):
        pass

    def test_get_path(self):
        pass
