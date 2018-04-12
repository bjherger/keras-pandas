import copy
import unittest

from auto_dl import lib
from auto_dl.Automater import Automater


class test_Automater(unittest.TestCase):

    def test_check_variable_lists(self):

        # Base case: No variables
        data = {
            'numerical_vars': [],
            'categorical_vars': [],
            'datetime_vars': []
        }
        self.assertEqual(True, lib.check_variable_list_are_valid(data))

        # Common use case: Variables in each
        data = {
            'numerical_vars': ['n1', 'n2', 'n3'],
            'categorical_vars': ['c1', 'c2', 'c3'],
            'datetime_vars': ['d1', 'd2']
        }
        self.assertEqual(True, lib.check_variable_list_are_valid(data))

        # Overlapping variable lists
        data = {
            'numerical_vars': ['n1', 'n2', 'n3', 'x1'],
            'categorical_vars': ['c1', 'c2', 'c3'],
            'datetime_vars': ['d1', 'd2', 'x1']
        }

        self.assertRaises(ValueError, lib.check_variable_list_are_valid, data)

        # Multiple overlapping variable lists
        data = {
            'numerical_vars': ['n1', 'n2', 'n3', 'x1'],
            'categorical_vars': ['c1', 'c2', 'c3', 'x1'],
            'datetime_vars': ['d1', 'd2', 'x1']
        }

        self.assertRaises(ValueError, lib.check_variable_list_are_valid, data)

    def test_Automater_initializer(self):

        # Base case: No variables
        auto = Automater()
        self.assertEqual({'numerical_vars': list(), 'categorical_vars': list(),
                                                    'boolean_vars': list(), 'datetime_vars': list(),
                                                    'non_transformed_vars': list()}, auto._variable_type_dict,)

        # Common use case: Variables in each
        data = {
            'numerical_vars': ['n1', 'n2', 'n3'],
            'categorical_vars': ['c1', 'c2', 'c3'],
            'datetime_vars': ['d1', 'd2']
        }

        response = copy.deepcopy(data)
        response['boolean_vars'] = list()
        response['non_transformed_vars'] = list()

        auto = Automater(numerical_vars=data['numerical_vars'], categorical_vars=data['categorical_vars'],
                         datetime_vars=data['datetime_vars'])

        self.assertEqual(response, auto._variable_type_dict)

        # Overlapping variable lists
        data = {
            'numerical_vars': ['n1', 'n2', 'n3', 'x1'],
            'categorical_vars': ['c1', 'c2', 'c3'],
            'datetime_vars': ['d1', 'd2', 'x1']
        }

        response = copy.deepcopy(data)
        response['boolean_vars'] = list()
        response['non_transformed_vars'] = list()

        self.assertRaises(ValueError, Automater().__init__(), numerical_vars=data['numerical_vars'],
                          categorical_vars=data['categorical_vars'],
                         datetime_vars=data['datetime_vars'])


