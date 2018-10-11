import logging

from keras import Model
from keras.layers import Dense

from keras_pandas import lib
from keras_pandas.Automater import Automater
from tests.testbase import TestBase


class TestNumeric(TestBase):

    def test_fit(self):
        iris_df = lib.load_iris()

        # One variable
        iris_numerical_cols = ['sepal_length']
        auto = Automater(numerical_vars=iris_numerical_cols)
        auto.fit(iris_df)

        self.assertEqual(Automater, type(auto))
        self.assertEqual(iris_numerical_cols, auto._user_provided_variables)
        self.assertTrue(auto.fitted)

        # Assert that transformation pipline has been built / trained
        self.assertEqual([['sepal_length']], list(map(lambda x: x[0], auto.input_mapper.built_features)))

    def test_transform(self):
        iris_df = lib.load_iris()

        # Two numerical variables, df_out = False
        iris_numerical_cols = ['sepal_length', 'sepal_width']
        auto = Automater(numerical_vars=iris_numerical_cols, df_out=False)
        auto.fit(iris_df)

        (X,y) = auto.transform(iris_df)
        self.assertEqual((150, ), X[0].shape)

        # Two numerical variables, df_out = True
        iris_numerical_cols = ['sepal_length', 'sepal_width']
        auto = Automater(numerical_vars=iris_numerical_cols, df_out=True)
        auto.fit(iris_df)

        transformed = auto.transform(iris_df)
        self.assertEqual(150, len(transformed.index))
        self.assertEqual((150, 2), transformed.shape)
        self.assertCountEqual(iris_numerical_cols, transformed.columns)

    def test_create_input_nub_numerical(self):
        iris_df = lib.load_iris()

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
        iris = lib.load_iris()
        iris_train = iris[:100]
        iris_test = iris[101:]
        iris_numerical_cols = ['sepal_length', 'petal_length']

        # Create auto
        auto = Automater(numerical_vars=iris_numerical_cols, response_var='sepal_length')

        # Train auto
        auto.fit(iris_train)
        X_train, y_train = auto.transform(iris_train)

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
        iris_test = iris_test.drop('sepal_length', axis=1)
        X_test, y_test = auto.transform(iris_test)
        model.predict(X_test)

        pass

