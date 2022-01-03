from unittest import TestCase
import numpy as np
import time
from tests import helpers
from dat_analysis.dat_object.attributes import logs
from dat_analysis.hdf_util import with_hdf_read
import json

output_dir = 'Outputs/Logs/'


class Testing_Logs(logs.Logs):
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
        pass

    def setUp(self):
        """Runs BEFORE every test"""
        self.Logs._init_sweeplogs()

    def test_0_init_sweeplogs(self):  # This has to run before all the others
        # Note: I run this before ALL tests anyway
        self.Logs._init_sweeplogs()
        self.assertIsInstance(self.Logs.sweeplogs, dict)

    def test_1__get_sweeplogs_from_exp(self):
        sweeplogs = self.Logs._get_sweeplogs_from_exp()
        self.assertIsInstance(sweeplogs, dict)

    def test_2__init_srss(self):
        self.Logs._init_srss()
        srss = self.Logs.srss
        self.assertIsInstance(srss, logs.SRSs)

    def test_3__init_babydac(self):
        self.Logs._init_babydac()
        bds = self.Logs.bds
        self.assertEqual(bds['CA0 offset'], -100)

    def test_4__init_fastdac(self):
        self.Logs._init_fastdac()
        fds = self.Logs.fds
        self.assertEqual(fds['LP*2'], -490.11)
        fastdac = self.Logs.Fastdac
        self.assertEqual(fastdac.measure_freq, 6060.6)
        self.assertEqual(fastdac.dacs, fds)

    def test_5__init_awg(self):
        self.Logs._init_awg()
        awg: logs.AWGtuple = self.Logs.awg
        # expected = Logs.AWGtuple(outputs={0: [0], 1: [1]}, wave_len=492, num_adcs=1, samplingFreq=6060.6,
        #                          measureFreq=6060.6, num_cycles=1, num_steps=148)
        expected = [{0: [0], 1: [1]}, 492, 1, 6060.6, 6060.6, 1, 148]
        self.assertEqual(expected,
                         [awg.outputs, awg.wave_len, awg.num_adcs, awg.samplingFreq, awg.measureFreq, awg.num_cycles,
                          awg.num_steps])

    def test_6__init_temps(self):
        self.Logs._init_temps()
        temps: logs.TEMPtuple = self.Logs.temps
        expected = [49.824, 3.8568, 4.671, 0.0499335, 0.66458]
        self.assertEqual(expected, [temps.fiftyk, temps.fourk, temps.mag, temps.mc, temps.still])

    def test_7__init_mags(self):
        self.Logs._init_mags()
        mags = self.Logs.mags
        expected = [-49.98, 'magy', 54.33]
        self.assertEqual(expected, [mags.magy.field, mags.magy.name, mags.magy.rate])

    def test_7a__init_other(self):
        self.Logs._init_other()
        key_expected = {
            'comments': 'transition, square entropy, repeat, ',
            'xlabel': 'LP*200 (mV)',
            'ylabel': 'Repeats',
            'measure_freq': 6060.6,
            'sampling_freq': 6060.6,
            'sweeprate': 49.939,
        }
        for k, v in key_expected.items():
            val = getattr(self.Logs, k)
            if isinstance(val, float):
                self.assertTrue(np.isclose(val, v, atol=0.001))
            else:
                self.assertEqual(v, val)

    def test_8__initialize_minimum(self):
        self.Logs.initialize_minimum()
        self.assertTrue(self.Logs.initialized)

    def test_9_assign_to_dat(self):
        self.dat.Logs = self.Logs
        self.assertEqual(self.dat.Logs, self.Logs)


class Test(TestCase):
    babydac_log_str = '{"DAC0{}":0, "DAC1{}":0, "DAC2{}":0, "DAC3{}":0, "DAC4{}":0, "DAC5{}":0, ' \
                      '"DAC6{}":0, "DAC7{CSbias/0.0001}":3000, "DAC8{RCB}":-150, "DAC9{RCT}":-150, ' \
                      '"DAC10{RCSQ}":-360, "DAC11{RCSS}":-360, "DAC12{CA0 offset}":-100, "DAC13{CA1 offset}":-193.1, ' \
                      '"DAC14{}":0, "DAC15{45}":0, "com_port":"ASRL6::INSTR"}'
    bd_dict = json.loads(babydac_log_str)

    def test__dac_logs_to_dict_with_BDs(self):
        bds = logs._dac_logs_to_dict(self.bd_dict)
        self.assertEqual(list(bds.keys())[0], 'DAC0')
        self.assertEqual(list(bds.keys())[6], 'DAC6')  # sort of checking order is maintained
        self.assertEqual(bds['CA0 offset'], -100)  # Checking some values match up
        self.assertEqual(bds['CSbias/0.0001'], 3000)
        self.assertEqual(len(bds), 16)  # Checking all here, and no extra keys

