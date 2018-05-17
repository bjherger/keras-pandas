import numpy

import pandas
import unittest

from keras import Model, losses
from keras.layers import Dense

from keras_pandas.Automater import Automater


class TestCategorical(unittest.TestCase):

    def test_fit(self):
        train_df = self.mushroom_dataframe()

        # Two variables
        mushroom_categorical_cols = ['odor', 'habitat']
        auto = Automater(categorical_vars=mushroom_categorical_cols)
        auto.fit(train_df)

        self.assertEqual(Automater, type(auto))
        self.assertEqual(mushroom_categorical_cols, auto._user_provided_variables)
        self.assertTrue(auto.fitted)

        # Assert that transformation pipline has been built / trained
        self.assertEqual([['odor'], ['habitat']], map(lambda x: x[0], auto.input_mapper.built_features))

    def test_transform_no_response(self):
        train_df = self.mushroom_dataframe()

        # Two numerical variables, df_out = False
        test_columns = ['odor', 'habitat']
        auto = Automater(categorical_vars=test_columns, df_out=False)
        auto.fit(train_df)

        (X,y) = auto.transform(train_df)
        self.assertEqual((8124, ), X[0].shape)
        self.assertEqual(None, y)

        # Two numerical variables, df_out = True
        test_columns = ['odor', 'habitat']
        auto = Automater(categorical_vars=test_columns, df_out=True)
        auto.fit(train_df)

        transformed = auto.transform(train_df)
        self.assertEqual(8124, len(transformed.index))
        self.assertEqual((8124, 2), transformed.shape)
        self.assertItemsEqual(test_columns, transformed.columns)

    def test_transform_with_response(self):
        train_df = self.mushroom_dataframe()

        # Two numerical variables, df_out = False
        test_columns = ['odor', 'habitat']
        auto = Automater(categorical_vars=test_columns, df_out=False, response_var='habitat')
        auto.fit(train_df)

        (X,y) = auto.transform(train_df)
        self.assertEqual((8124, ), X[0].shape)

        # Two numerical variables, df_out = True
        test_columns = ['odor', 'habitat']
        auto = Automater(categorical_vars=test_columns, df_out=True, response_var='habitat')
        auto.fit(train_df)

        transformed = auto.transform(train_df)
        self.assertEqual(8124, len(transformed.index))
        self.assertEqual((8124, 2), transformed.shape)
        self.assertItemsEqual(test_columns, transformed.columns)

        # Test w/ response var unavailable.
        test_columns = ['odor']
        test_df = train_df[test_columns]
        transformed = auto.transform(test_df)
        self.assertEqual(8124, len(transformed.index))
        self.assertEqual((8124, 1), transformed.shape)
        self.assertItemsEqual(test_columns, transformed.columns)

    def test_create_input_nub_numerical(self):
        train_df = self.mushroom_dataframe()

        # Zero variables
        variable_type_dict = {'categorical_vars': []}
        input_layers, input_nub = Automater()._create_input_nub(variable_type_dict, train_df)
        self.assertEqual(list(), input_layers)

        # One variable
        iris_numerical_cols = ['odor']
        variable_type_dict = {'numerical_vars': iris_numerical_cols}
        input_layers, input_nub = Automater(numerical_vars=iris_numerical_cols).\
            _create_input_nub(variable_type_dict, train_df)
        # TODO Check layer type
        self.assertEqual(1, len(input_layers))

        # Multiple numeric variables
        iris_numerical_cols = ['odor', 'habitat', 'population']
        variable_type_dict = {'numerical_vars': iris_numerical_cols}
        input_layers, input_nub = Automater(numerical_vars=iris_numerical_cols).\
            _create_input_nub(variable_type_dict, train_df)
        self.assertEqual(3, len(input_layers))

    def test_numerical_whole(self):
        # St up data set
        mushroom_df = self.mushroom_dataframe()
        msk = numpy.random.rand(len(mushroom_df)) < 0.95
        mushroom_train = mushroom_df[msk]
        mushroom_test = mushroom_df[~msk]
        iris_numerical_cols = ['odor', 'habitat', 'population', 'class']

        # Create auto
        auto = Automater(categorical_vars=iris_numerical_cols, response_var='class')

        # Train auto
        auto.fit(mushroom_train)
        X_train, y_train = auto.transform(mushroom_train)

        # Extract input_nub from auto
        input_nub = auto.input_nub

        # Extract output_nub from auto
        output_nub = auto.output_nub

        # Create DL model
        x = input_nub
        x = Dense(30)(x)
        x = output_nub(x)

        model = Model(inputs=auto.input_layers, outputs=x)
        model.compile(optimizer='Adam', loss=losses.sparse_categorical_crossentropy)

        # Train DL model
        model.fit(X_train, y_train)

        # Transform test set
        mushroom_test = mushroom_test.drop('class', axis=1)
        X_test, y_test = auto.transform(mushroom_test)
        model.predict(X_test)

        pass

    @staticmethod
    def mushroom_dataframe():
        return pandas.read_csv('test_data/mushrooms.csv')
