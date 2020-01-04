from unittest import TestCase
from tests import helpers as th
th.Dirs.set_test_dirs()

from src.DFcode import SetupDF
from src import config as cfg
import os
import pandas as pd
from src.DFcode.DFutil import protect_data_from_reindex
th.setverbose(False, 20)

pd.DataFrame.reset_index = protect_data_from_reindex(pd.DataFrame.reset_index)


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
        _data = {'datetime': 'Wednesday, January 1, 2020 00:00:00', 'datnumplus': 2650, 'FastScanCh0': 1e8, 'FastScanCh1': 5, 'fd_0adc': 1e8}
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
        setupdf.df.reset_index(['i_sense'])
        self.assertEqual(True, setupdf.loaded)  # Assert setupdf loaded from pkl

    def test_add_row_noconflict(self):
        _data = {'FastScanCh0': 1e7, 'FastScanCh1': None}
        _datetime = 'Thursday, January 2, 2020 12:00:00'
        _datnum = 2680
        setupdf = TestSetupDF.getfilledDF()
        setupdf.add_row(_datetime, _datnum, _data)

        self.assertEqual(1e8, setupdf.df.loc[2650, 'FastScanCh0'])
        self.assertEqual(1e7, setupdf.df.loc[2680, 'FastScanCh0'])
        self.assertEqual(1e7, setupdf.df.loc[2650, 'FastScanCh1'])
        self.assertEqual(None, setupdf.df.loc[2680, 'FastScanCh1'])
        return None
