from keras_pandas.Automater import Automater
from tests.testbase import TestBase


class TestAutomater(TestBase):

    def test_init(self):
        # Supervised

        variable_type_dict = {'numerical': ['n1', 'n2'],
                              'categorical': ['c1', 'c2'],
                              'boolean': ['b1']}

        auto = Automater(variable_type_dict=variable_type_dict,
                         response_var='c1')

        self.assertTrue(auto.supervised)
        self.assertCountEqual(['n1', 'n2', 'c2', 'b1'], auto.input_vars)
        self.assertEqual('c1', auto.response_var)
        self.assertIsNone(auto.input_mapper)
        self.assertIsNone(auto.output_mapper)
        self.assertFalse(auto.fitted)

        # Unsupervised
        auto = Automater(variable_type_dict=variable_type_dict)
        self.assertFalse(auto.supervised)
        self.assertCountEqual(['n1', 'n2', 'c1', 'c2', 'b1'], auto.input_vars)
        self.assertEqual(None, auto.response_var)
        self.assertIsNone(auto.input_mapper)
        self.assertIsNone(auto.output_mapper)
        self.assertFalse(auto.fitted)

        pass

    def test_fit(self):
        pass

    def test_transform(self):
        pass

    def test_fit_transform(self):
        pass

    def test__create_input_nub(self):
        pass

    def test_create_output_nub(self):
        pass

    def test_create_mapper(self):
        pass

    def test_check_response_var(self):
        pass

    def test_check_input_df(self):
        pass

    def test_valid_configurations_check(self):
        pass


