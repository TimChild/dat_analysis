from unittest import TestCase
from tests import helpers
from src.DatObject.Attributes import Logs
from src.HDF_Util import with_hdf_read
import json
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

    def tearDown(self):
        """Runs AFTER every test"""
        with self.assertRaises(ValueError):
            self.dat.hdf.hdf.filename  # Checking hdf is actually closed

    def setUp(self):
        """Runs BEFORE every test"""
        self.Logs._init_sweeplogs()

    def test_init_sweeplogs(self):  # This has to run before all the others
        # Note: I run this before ALL tests anyway
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
        bds = self.Logs.bds
        self.assertEqual(bds['CA0 offset'], -100)

    def test__init_fastdac(self):
        self.Logs._init_fastdac()
        fds = self.Logs.fds
        self.assertEqual(fds['LP*2'], -490.11)
        fastdac = self.Logs.Fastdac
        self.assertEqual(fastdac.measure_freq, 6060.6)
        self.assertEqual(fastdac.dacs, fds)

    def test__init_awg(self):
        self.fail()

    def test__init_temps(self):
        self.fail()

    def test__init_mags(self):
        self.fail()

    def test__initialize_minimum(self):
        self.assertTrue(self.Logs.initialized)
        self.fail()


    def test_assign_to_dat(self):
        self.dat.Logs = Logs
        self.assertEqual(self.dat.Logs, Logs)


class Test(TestCase):
    babydac_log_str = '{"DAC0{}":0, "DAC1{}":0, "DAC2{}":0, "DAC3{}":0, "DAC4{}":0, "DAC5{}":0, ' \
                      '"DAC6{}":0, "DAC7{CSbias/0.0001}":3000, "DAC8{RCB}":-150, "DAC9{RCT}":-150, ' \
                      '"DAC10{RCSQ}":-360, "DAC11{RCSS}":-360, "DAC12{CA0 offset}":-100, "DAC13{CA1 offset}":-193.1, ' \
                      '"DAC14{}":0, "DAC15{45}":0, "com_port":"ASRL6::INSTR"}'
    bd_dict = json.loads(babydac_log_str)

    def test__dac_logs_to_dict_with_BDs(self):
        bds = Logs._dac_logs_to_dict(self.bd_dict)
        self.assertEqual(list(bds.keys())[0], 'DAC0')
        self.assertEqual(list(bds.keys())[6], 'DAC6')  # sort of checking order is maintained
        self.assertEqual(bds['CA0 offset'], -100)  # Checking some values match up
        self.assertEqual(bds['CSbias/0.0001'], 3000)
        self.assertEqual(len(bds), 16)  # Checking all here, and no extra keys


