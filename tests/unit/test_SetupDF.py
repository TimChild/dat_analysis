from unittest import TestCase
from tests import helpers as th
from src.DFcode import SetupDF
from src.Configs import Main_Config as cfg
import os
import pandas as pd
from src.DFcode.DFutil import protect_data_from_reindex
th.setverbose(False, 20)
th.Dirs.set_test_dirs()
pd.DataFrame.set_index = protect_data_from_reindex(pd.DataFrame.set_index)


class TestSetupDF(TestCase):
    setupdir = os.path.join(cfg.dfdir, 'setup/')

    @staticmethod
    def killDFsaves():
        SetupDF.SetupDF.killinstance()
        for savetype in ['pkl', 'xlsx']:
            if os.path.isfile(f'{TestSetupDF.setupdir}/setup.{savetype}'):
                os.remove(f'{TestSetupDF.setupdir}/setup.{savetype}')

    @staticmethod
    def getfilledDF():
        _data = {'datetime': 'Wednesday, January 1, 2020 00:00:00', 'datnumplus': 2650, 'FastScanCh0': 1e8, 'FastScanCh1': 5, 'FastScanCh2': 10}
        TestSetupDF.killDFsaves()
        setupdf = SetupDF.SetupDF()
        setupdf.add_row(_data['datetime'], _data['datnumplus'], dict([item for item in _data.items() if item[0] not in ['datetime', 'datnumplus']]))
        return setupdf

    def test_save(self):
        TestSetupDF.killDFsaves()  # enforces no instance or saves to start
        setupdf = SetupDF.SetupDF()
        setupdf.save()
        for savetype in ['pkl', 'xlsx']:
            self.assertEqual(True, os.path.isfile(os.path.join(TestSetupDF.setupdir, f'setup.{savetype}')))
        return

    def test_load(self):
        TestSetupDF.test_save(self)  # ensures saved pkl should exist
        SetupDF.SetupDF.killinstance()
        setupdf = SetupDF.SetupDF()
        self.assertEqual(True, setupdf.loaded)  # Assert setupdf loaded from pkl

    def test_add_row_noconflict(self):
        _data = {'FastScanCh0': 1e7, 'FastScanCh1': None}
        _datetime = 'Thursday, January 2, 2020 12:00:00'
        _datnum = 2680
        setupdf = TestSetupDF.getfilledDF()
        setupdf.add_row(_datetime, _datnum, _data)
        setupdf.df.reset_index()
        setupdf.df.set_index(['datnumplus'], inplace=True)
        self.assertEqual(1e8, setupdf.df.loc[2650, 'FastScanCh0'])
        self.assertEqual(1e7, setupdf.df.loc[2680, 'FastScanCh0'])
        self.assertEqual(5, setupdf.df.loc[2650, 'FastScanCh1'])
        self.assertEqual(True, pd.isna(setupdf.df.loc[2680, 'FastScanCh1']))
        return None

