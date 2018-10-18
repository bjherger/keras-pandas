import numpy

import pandas
import unittest

from keras import Model, losses
from keras.layers import Dense

from keras_pandas import lib
from keras_pandas.Automater import Automater
from tests.testbase import TestBase


class TestCategorical(TestBase):

    def test_fit(self):
        train_df = lib.load_mushroom()

        # Two variables
        mushroom_categorical_cols = ['odor', 'habitat']
        auto = Automater(categorical_vars=mushroom_categorical_cols)
        auto.fit(train_df)

        self.assertEqual(Automater, type(auto))
        self.assertEqual(mushroom_categorical_cols, auto._user_provided_variables)
        self.assertTrue(auto.fitted)

        # Assert that transformation pipline has been built / trained
        self.assertEqual([['odor'], ['habitat']], list(map(lambda x: x[0], auto.input_mapper.built_features)))

    def test_transform_no_response(self):
        train_df = lib.load_mushroom()

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
        self.assertCountEqual(test_columns, transformed.columns)

    def test_transform_with_response(self):
        train_df = lib.load_mushroom()

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
        self.assertCountEqual(test_columns, transformed.columns)

        # Test w/ response var unavailable.
        test_columns = ['odor']
        test_df = train_df[test_columns]
        transformed = auto.transform(test_df)
        self.assertEqual(8124, len(transformed.index))
        self.assertEqual((8124, 1), transformed.shape)
        self.assertCountEqual(test_columns, transformed.columns)

    def test_create_input_nub_numerical(self):
        # TODO rename function, there is no numerical input
        train_df = lib.load_mushroom()

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

    def test_boolean(self):
        observations = lib.load_mushroom()
        observations['population_bool'] = observations['population'] == 's'

        msk = numpy.random.rand(len(observations)) < 0.95
        mushroom_train = observations[msk]
        mushroom_test = observations[~msk]

        categorical_vars = ['odor', 'habitat', 'class']
        boolean_vars = ['population_bool']

        auto = Automater(categorical_vars=categorical_vars, boolean_vars=boolean_vars,  response_var='class')

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
        model.compile(optimizer='Adam', loss=auto.loss)

        # Train DL model
        model.fit(X_train, y_train)

        # Transform test set
        mushroom_test = mushroom_test.drop('class', axis=1)
        X_test, y_test = auto.transform(mushroom_test)
        model.predict(X_test)


    def test_categorical_whole(self):
        # St up data set
        mushroom_df = lib.load_mushroom()
        msk = numpy.random.rand(len(mushroom_df)) < 0.95
        mushroom_train = mushroom_df[msk]
        mushroom_test = mushroom_df[~msk]
        categorical_vars = ['odor', 'habitat', 'population', 'class']

        # Create auto
        auto = Automater(categorical_vars=categorical_vars, response_var='class')

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
        model.compile(optimizer='Adam', loss=auto.loss)

        # Train DL model
        model.fit(X_train, y_train)

        # Transform test set
        mushroom_test = mushroom_test.drop('class', axis=1)
        X_test, y_test = auto.transform(mushroom_test)
        model.predict(X_test)

        pass

