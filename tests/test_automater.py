import copy
import logging
import unittest

import pandas
from keras import Model, losses
from keras.layers import Dense

from auto_dl import lib
from auto_dl.Automater import Automater

logging.getLogger().setLevel(logging.DEBUG)


class test_automater(unittest.TestCase):

    def test_check_variable_lists_are_valid(self):
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

    def test_create_sklearn_pandas_mapper_pipeline_length(self):
        # Base case: No variables
        data = {}
        mapper = Automater()._create_sklearn_pandas_mapper(data)
        self.assertItemsEqual(list(), mapper.features)

        # A single numerical
        data = {'numerical_vars': ['n1']}
        mapper = Automater()._create_sklearn_pandas_mapper(data)
        self.assertEqual(1, len(mapper.features))

        # Two numerical
        data = {'numerical_vars': ['n1', 'n2']}
        mapper = Automater()._create_sklearn_pandas_mapper(data)
        self.assertEqual(2, len(mapper.features))

        # Two variables of different types
        data = {'numerical_vars': ['n1'],
                'categorical_vars': ['c1']}
        mapper = Automater()._create_sklearn_pandas_mapper(data)
        self.assertEqual(2, len(mapper.features))

        # Two varibles with default pipelines
        data = {'NO_DEFAULT_ASDFSDA': ['x1', 'x2']}
        mapper = Automater()._create_sklearn_pandas_mapper(data)
        self.assertEqual(2, len(mapper.features))

        mapper_pipelines = map(lambda x: x[1], mapper.features)

        self.assertItemsEqual([None, None], mapper_pipelines)

    def test_initializer(self):
        # Base case: No variables
        auto = Automater()
        self.assertEqual({'numerical_vars': list(), 'categorical_vars': list(),
                          'boolean_vars': list(), 'datetime_vars': list(),
                          'non_transformed_vars': list()}, auto._variable_type_dict, )
        self.assertItemsEqual(list(), auto._user_provided_input_variables)

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

        self.assertEqual(False, auto.fitted)
        self.assertEqual(response, auto._variable_type_dict)

        response_variable_list = [item for sublist in response.values() for item in sublist]
        self.assertItemsEqual(response_variable_list, auto._user_provided_input_variables)

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

        # TODO Test that df_out is captured correctly

    def test_fit(self):
        iris_df = self.iris_dataframe()

        # One variable
        iris_numerical_cols = ['sepal_length']
        auto = Automater(numerical_vars=iris_numerical_cols)
        auto.fit(iris_df)

        self.assertEqual(Automater, type(auto))
        self.assertEqual(iris_numerical_cols, auto._user_provided_input_variables)
        self.assertTrue(auto.fitted)

        # Assert that transformation pipline has been built / trained
        self.assertEqual([['sepal_length']], map(lambda x: x[0], auto._sklearn_pandas_mapper.built_features))

    def test_transform(self):
        iris_df = self.iris_dataframe()

        # Two numerical variables, df_out = False
        iris_numerical_cols = ['sepal_length', 'sepal_width']
        auto = Automater(numerical_vars=iris_numerical_cols, df_out=False)
        auto.fit(iris_df)

        transformed = auto.transform(iris_df)
        self.assertEqual((150, 2), transformed[0].shape)

        # Two numerical variables, df_out = True
        iris_numerical_cols = ['sepal_length', 'sepal_width']
        auto = Automater(numerical_vars=iris_numerical_cols, df_out=True)
        auto.fit(iris_df)

        transformed = auto.transform(iris_df)
        self.assertEqual(150, len(transformed.index))
        self.assertEqual((150, 2), transformed.shape)
        self.assertItemsEqual(['sepal_length', 'sepal_width'], transformed.columns)

    def test_create_input_nub_numerical(self):
        iris_df = self.iris_dataframe()

        # Zero variables
        variable_type_dict = {'numerical_vars': []}
        input_layers, input_nub = Automater()._create_input_nub(variable_type_dict, iris_df)
        self.assertEqual(list(), input_layers)

        # One variable
        iris_numerical_cols = ['sepal_length']
        variable_type_dict = {'numerical_vars': iris_numerical_cols}
        input_layers, input_nub = Automater(numerical_vars=iris_numerical_cols)._create_input_nub(variable_type_dict,
                                                                                                  iris_df)
        self.assertEqual(1, len(input_layers))

        # Multiple numeric variables
        iris_numerical_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        variable_type_dict = {'numerical_vars': iris_numerical_cols}
        input_layers, input_nub = Automater(numerical_vars=iris_numerical_cols)._create_input_nub(variable_type_dict,
                                                                                                  iris_df)
        self.assertEqual(4, len(input_layers))

    def test_numerical_whole(self):
        # St up data set
        iris = self.iris_dataframe()
        iris_numerical_cols = ['sepal_length', 'petal_length']

        # Create auto
        auto = Automater(numerical_vars=iris_numerical_cols, response_var='sepal_length')

        # Train auto
        auto.fit(iris, y='sepal_length')
        X, y = auto.transform(iris)

        # Extract input_nub from auto
        input_nub = auto.input_nub

        # Extract output_nub from auto
        output_nub = auto.output_nub

        # Create DL model
        x = input_nub
        x = Dense(30)(x)
        x = output_nub(x)

        model = Model(inputs=auto.input_layers, outputs=x)
        model.compile(optimizer='Adam', loss=losses.mean_squared_error)

        # Train DL model
        model.fit(X, y)

        pass

    @staticmethod
    def iris_dataframe():
        return pandas.read_csv('test_data/iris.csv')
