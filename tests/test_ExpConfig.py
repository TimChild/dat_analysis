from unittest import TestCase
import dat_analysis.hdf_util as HDU
import h5py
from dat_analysis.hdf_util import with_hdf_read
from dat_analysis.data_standardize.exp_config import ExpConfigBase, ExpConfigGroupDatAttribute, DataInfo
from tests.helpers import get_testing_ExpConfig, init_testing_dat
from tests import helpers
from dat_analysis.dat_object.dat_hdf import DatHDFBuilder

output_dir = 'Outputs/ExpConfig/'
dat_dir = 'fixtures/dats/2020Sep/'


class Testing_ExpConfigDatAttribute(ExpConfigGroupDatAttribute):
    @with_hdf_read
    def check_init(self):
        group = self.hdf.get(self.group_name, None)
        if group is None:
            self._create_group(self.group_name)
            group = self.hdf.get(self.group_name)
        if group.attrs.get('initialized', False) is False:
            # self._initialize()  # This will run everything otherwise
            pass


ExpConfig = get_testing_ExpConfig()


class Test_ExpConfig(TestCase):
    pass


class TestExpConfigGroupDatAttribute(TestCase):
    helpers.clear_outputs(output_dir)
    exp2hdf = helpers.get_testing_Exp2HDF(dat_dir=dat_dir, output_dir=output_dir)(9111)
    builder = DatHDFBuilder(exp2hdf, init_level='min')
    builder.create_hdf()
    builder.copy_exp_data()
    builder.init_DatHDF()
    builder.init_base_attrs()
    dat = builder.dat
    E = Testing_ExpConfigDatAttribute(dat, exp_config=ExpConfig(9111))

    def setUp(self):
        """Runs before every test"""
        pass

    def tearDown(self):
        """Runs after every test"""
        pass

    def test__set_default_data_descriptors(self):
        self.E._set_default_data_descriptors()
        with h5py.File(self.E.hdf.hdf_path, 'r') as f:
            group = f.get(self.E.group_name + '/Default DataDescriptors')
            keys = group.keys()
            self.assertTrue({'cscurrent', 'cscurrent_2d', 'x_array', 'y_array'} -
                            set(keys) == set())  # Check expected keys are there

    def test__initialize_minimum(self):
        self.E.initialize_minimum()
        self.assertTrue(self.E.initialized)

    def test__set_sweeplog_subs(self):
        self.E._set_sweeplog_subs()
        with h5py.File(self.E.hdf.hdf_path, 'r') as f:
            group = f.get(self.E.group_name)
            subs = HDU.get_attr(group, 'sweeplog_substitutions', None)
            self.assertEqual(subs, {'FastDAC 1': 'FastDAC'})

    def test_get_sweeplogs(self):
        sweeplogs = self.E.get_sweeplogs()
        self.assertEqual(sweeplogs['filenum'], 9111)

    def test_get_default_data_infos(self):
        self.test__set_default_data_descriptors()
        default_infos = self.E.get_default_data_infos()
        expected_info = DataInfo('i_sense')
        self.assertEqual(default_infos['cscurrent'], expected_info)

    def test_clear_caches(self):
        self.E.clear_caches()
        self.assertTrue(True)
