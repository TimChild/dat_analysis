from unittest import TestCase
from tests import helpers
from src.DatObject.Attributes import Logs
from src.HDF_Util import with_hdf_read

output_dir = 'Outputs/Logs/'


class Testing_Logs(Logs.Logs):
    """Override the normal init behaviour so it doesn't fail before reaching tests"""
    @with_hdf_read
    def check_init(self):
        group = self.hdf.get(self.group_name, None)
        if group is None:
            self._create_group(self.group_name)
            group = self.hdf.get(self.group_name)
        if group.attrs.get('initialized', False) is False:
            # self._initialize()  # This will run everything otherwise
            pass


class TestLogs(TestCase):
    """Note: Many of these tests depend on dat.ExpConfig working well, so look there if these fail too."""
    # Initial Setup
    helpers.clear_outputs(output_dir)
    dat = helpers.init_testing_dat(9111, output_dir)
    Logs = Testing_Logs(dat)  # This should make Logs without running init, and will not assign it to the dat yet

    def test_0_init_sweeplogs(self):  # This has to run before all the others
        self.Logs._init_sweeplogs()
        self.assertIsInstance(self.Logs.sweeplogs, dict)

    def test__get_sweeplogs_from_exp(self):
        sweeplogs = self.Logs._get_sweeplogs_from_exp()
        self.assertIsInstance(sweeplogs, dict)

    def test__init_srss(self):
        self.Logs._init_srss()
        srss = self.Logs.srss
        self.assertIsInstance(srss, Logs.SRSs)

    def test__init_babydac(self):
        self.Logs._init_babydac()
        bds = self.Logs.babydacs
        self.fail()

    def test__init_fastdac(self):
        self.fail()

    def test__init_awg(self):
        self.fail()

    def test__init_temps(self):
        self.fail()

    def test__init_mags(self):
        self.fail()

    def test__initialize_minimum(self):
        self.fail()

    def test_assign_to_dat(self):
        dat.Logs = Logs
        self.assertEqual(dat.Logs, Logs)