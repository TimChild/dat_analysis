from unittest import TestCase
from tests import helpers as th
th.Dirs.set_test_dirs()

from src.DFcode import SetupDF
from src import config as cfg
import os

th.setverbose(False, 20)


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
        _data = {'datnumplus': 2650, 'BOB1': 1e8, }
        TestSetupDF.killDFsaves()
        setupdf = SetupDF.SetupDF()


    def test_save(self):
        TestSetupDF.killDFsaves()  # enforces no instance or saves to start
        setupdf = SetupDF.SetupDF()
        setupdf.save()
        for savetype in ['pkl', 'xlsx']:
            self.assertEqual(True, os.path.isfile(os.path.join(TestSetupDF.setupdir, f'setup.{savetype}')))
