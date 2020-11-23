from unittest import TestCase
from src.DatObject.Attributes import Transition
from tests import helpers
from src.HDF_Util import with_hdf_read

output_dir = 'Outputs/DatAttribute/'


class Testing_Transition(Transition.Transition):
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


class TestDatAttributeWithData(TestCase):
    helpers.clear_outputs(output_dir)
    dat = helpers.init_testing_dat(9111, output_dir)
    T: Transition.Transition = Testing_Transition(dat)

    def test_get_data(self):
        self.fail()

    def test__get_data(self):
        self.fail()

    def test_set_data(self):
        self.fail()

    def test__set_data(self):
        self.fail()

    def test_specific_data_descriptors(self):
        self.fail()

    def test_set_data_descriptor(self):
        self.fail()

    def test_get_descriptor(self):
        self.fail()


class TestFittingAttribute(TestCase):
    helpers.clear_outputs(output_dir)
    dat = helpers.init_testing_dat(9111, output_dir)
    T: Transition.Transition = Testing_Transition(dat)

    def test_default_data_name(self):
        self.fail()

    def test_get_default_params(self):
        self.fail()

    def test_get_default_func(self):
        self.fail()

    def test_default_data_names(self):
        self.fail()

    def test_clear_caches(self):
        self.fail()

    def test_get_centers(self):
        self.fail()

    def test_get_data(self):
        self.fail()

    def test_set_data(self):
        self.fail()

    def test_avg_data(self):
        self.fail()

    def test_avg_x(self):
        self.fail()

    def test_avg_data_std(self):
        self.fail()

    def test_avg_fit(self):
        self.fail()

    def test_row_fits(self):
        self.fail()

    def test_get_avg_data(self):
        self.fail()

    def test__make_avg_data(self):
        self.fail()

    def test_get_fit(self):
        self.fail()

    def test__get_fit_from_path(self):
        self.fail()

    def test__get_fit_paths(self):
        self.fail()

    def test__get_fit_path_from_fit_id(self):
        self.fail()

    def test__get_fit_path_from_name(self):
        self.fail()

    def test__generate_fit_path(self):
        self.fail()

    def test__save_fit(self):
        self.fail()

    def test__get_fit_parent_group_name(self):
        self.fail()

    def test__calculate_fit(self):
        self.fail()

    def test_initialize_minimum(self):
        self.fail()

    def test_set_default_data_descriptors(self):
        self.fail()
