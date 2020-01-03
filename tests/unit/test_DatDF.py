from unittest import TestCase

import pandas as pd

import src.CoreUtil
import src.DFcode.DatDF
from src.DFcode.DatDF import savetodf
from src import ExampleExperimentSpecific as EES
import tests.helpers as th
import os
from unittest.mock import patch
import src.config as cfg

th.Dirs.set_test_dirs()
th.setverbose(False, level=25)


class TestDatDF(TestCase):

    @staticmethod
    def makefilledDF(name):
        """Makes DF with some information. Maybe want to put this somewhere else if it gets bigger"""
        datDF = src.DFcode.DatDF.DatDF(dfname=name)
        inputs = ['y', 'y', 'y', 'y', 'y', 'y']
        with patch('builtins.input', side_effect=th.simple_mock_input(inputs)) as mock:
            dat = EES.make_dat_standard(2700, dfname=name)
        savetodf(dat, dfname=name)

    @staticmethod
    def makeDFstandalone():
        _default_columns = ['time', 'picklepath', 'x_label', 'y_label', 'dim', 'time_elapsed']
        _default_data = [['Wednesday, January 1, 2020 00:00:00', 'pathtopickle', 'xlabel', 'ylabel', 1, 1]]
        _dtypes = ['datetime', str, str, str, float, float]
        _dtypes = dict(zip(_default_columns, _dtypes))  # puts into form DataFrame can use

        def set_dtypes(df):
            """Sets correct dtypes for each column"""
            for key, value in _dtypes.items():
                if type(value) == type:
                    df[key] = df[key].astype(value)
                elif value == 'datetime':
                    df[key] = pd.to_datetime(df[key])

        mux = pd.MultiIndex.from_arrays([[0], ['base']],
                                        names=['datnum', 'datname'])  # Needs at least one row of data to save
        df = pd.DataFrame(src.DFcode.DatDF.DatDF._default_data, index=mux, columns=_default_columns)
        set_dtypes(df)
        return df

    @staticmethod
    def getfilledDF(name):
        # if not os.path.isfile(f'{cfg.dfdir}/{name}.pkl'):  #TODO: make this load a premade df instead of rebuilding each time
        TestDatDF.killDF(name)
        TestDatDF.makefilledDF(name)
        return src.DFcode.DatDF.DatDF(dfname=name)

    @staticmethod
    def killDF(name):
        src.DFcode.DatDF.DatDF.killinstance(name)
        for savetype in ['pkl', 'xlsx']:
            if os.path.isfile(f'{cfg.dfdir}/{name}.{savetype}'):
                os.remove(f'{cfg.dfdir}/{name}.{savetype}')

    @staticmethod
    def getcleanDF(name):
        TestDatDF.killDF(name)
        return src.DFcode.DatDF.DatDF(dfname=name)

    def test_newDF(self):  # Removes then creates new empty DF
        """Just tests that creating a new dat makes a pkl and xlsx file"""
        datDF = TestDatDF.getcleanDF('new')
        self.assertEqual(datDF.loaded, False)
        self.assertTrue(os.path.isfile(f'{cfg.dfdir}/new.pkl'))
        self.assertTrue(os.path.isfile(f'{cfg.dfdir}/new.xlsx'))

    def test_add_dat(self):
        """Adding row of dat values to DF (this will need modifying when more info is added to dats)"""
        datDF = TestDatDF.getcleanDF('test_add')
        inputs = ['y', 'y', 'y', 'y', 'y', 'y']
        with patch('builtins.input', side_effect=th.simple_mock_input(inputs)) as mock:
            dat = EES.make_dat_standard(2700, dfname='test_add')
        inputs = ['y', 'y', 'y', 'y', 'y']  # Adding new columns to clean DF
        with patch('builtins.input', side_effect=th.simple_mock_input(inputs)) as mock:
            savetodf(dat, dfname='test_add')

    def test_add_dat_attr(self):
        """Make sure correct things are rejected/allowed to be entered in DF"""
        datDF = TestDatDF.getcleanDF('test_add_attr')
        for col, val in zip(['stringcol', 'intcol', 'floatcol'], ['string', '5', '5.4']):
            with patch('builtins.input', return_value='y'):  # Just enters yes when asked to add new columns
                datDF.add_dat_attr(datnum=1, attrname=col, attrvalue=val, datname='base')
            self.assertEqual(datDF.df.at[(1, 'base'), col], val)
        import numpy as np
        for col, val in zip(['datnum', 'dfname', 'array', 'dict'], [1, 'name', np.array([1, 1]), {'a': 1}]):
            datDF.add_dat_attr(datnum=1, attrname=col, attrvalue=val, datname='base')
            with self.assertRaises(KeyError):
                print(datDF.df.at[(1, 'base'), col])

    def test_save_load(self):
        """Tests that DF saves correctly, Tests if loading without killing open DF opens same instance, 
        Tests if loading from file give same df"""

        datDF = TestDatDF.getfilledDF('test_save_load')
        TestDatDF.killDF('save')
        datDF.save('save')
        for ext in ['pkl', 'xlsx']:
            self.assertTrue(os.path.isfile(f'{cfg.dfdir}/save.{ext}'))

        datDF2 = src.DFcode.DatDF.DatDF(dfname='save')
        self.assertEqual(datDF, datDF2)  # Should get same instance and not load from file

        src.DFcode.DatDF.DatDF.killinstances()
        datDF3 = src.DFcode.DatDF.DatDF(dfname='save')
        self.assertTrue(datDF.df.equals(datDF3.df))  # should have to load from file

    def test_check_dtype(self):
        df = TestDatDF.makeDFstandalone()
        for attrname, attrvalue in zip(['dim', 'x_label'], [2, 'testlabel']):  # both correct dtypes
            self.assertTrue(src.DFcode.DatDF.DatDF.check_dtype(df, attrname, attrvalue))
        for ans, truth in zip(['n', 'y'], [False, True]):
            for attrname, attrvalue in zip(['dim', 'x_label'], ['2', 1]):  # Both wrong dtypes
                with patch('builtins.input',
                           side_effect=th.simple_mock_input(ans)):  # Check returns False when user says no don't change, and True when user says go ahead
                    self.assertEqual(src.DFcode.DatDF.DatDF.check_dtype(df, attrname, attrvalue), truth)

    def test_sync_dat(self):
        pass

    def test_get_path(self):
        pass

    def test_load(self):
        datdf = TestDatDF.getfilledDF('test_load')

        datdf.df.loc[(0, 'base'), 'x_label'] = 'new_xlabel'         #Emulate external change
        datdf.df.to_excel(f'{cfg.dfdir}/test_load.xlsx')  ##

        with patch('builtins.input', side_effect=th.simple_mock_input(['y', 'n'])) as mock: # Do you want to load excel version?
            datdf = datdf.load()  # Should load new version
            self.assertEqual(datdf.df.loc[(0, 'base'), 'x_label'], 'new_xlabel')
            datdf = datdf.load()  # Should load old version this time (because no save was done)
            self.assertEqual(datdf.df.loc[(0, 'base'), 'x_label'], 'xlabel')

    def test_infodict(self):
        """Tests whether get infodict from df runs"""
        df = TestDatDF.getfilledDF('test_infodict')
        infodict = df.infodict(2700, 'base')



            
class TestAddColLabel(TestCase):
    dfdefault = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['one', 'two', 'three'])

    def test_add_new_level1(self):
        df = TestAddColLabel.dfdefault
        df2 = src.CoreUtil.add_col_label(df, '2nd', ['two', 'three'])
        self.assertEqual(['', '2nd', '2nd'], [x[1] for x in df2.columns])

    def test_write_overwrite_existing_level(self):
        df = TestAddColLabel.dfdefault
        df2 = src.CoreUtil.add_col_label(df, '2nd', ['two', 'three'])
        df3 = src.CoreUtil.add_col_label(df2, '3rd', ['one', 'three'], level=1)
        self.assertEqual(['3rd', '2nd', '3rd'], [x[1] for x in df3.columns])

    def test_add_new_level2(self):
        df = TestAddColLabel.dfdefault
        df2 = src.CoreUtil.add_col_label(df, '2nd', ['two', 'three'])
        df4 = src.CoreUtil.add_col_label(df2, '4th', ['one', 'three'], level=2)
        self.assertEqual(['4th', '', '4th'], [x[2] for x in df4.columns])

    def test_full_address(self):
        df = TestAddColLabel.dfdefault
        df2 = src.CoreUtil.add_col_label(df, '2nd', ['two', 'three'])
        df3 = src.CoreUtil.add_col_label(df2, '3rd', [('three', '2nd')], level=2)
        self.assertEqual(['', '', '3rd'], [x[2] for x in df3.columns])

