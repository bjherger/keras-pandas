import logging
import unittest

import numpy
import pandas
from keras import losses, Model
from keras.layers import Dense

from keras_pandas import lib
from keras_pandas.Automater import Automater
from tests.testbase import TestBase


class TestText(TestBase):

    def test_fit(self):
        data = lib.load_titanic()
        # One variable
        text_vars = ['name']

        auto = Automater(text_vars=text_vars)
        auto.fit(data)

        self.assertEqual(Automater, type(auto))
        self.assertEqual(text_vars, auto._user_provided_variables)
        self.assertTrue(auto.fitted)

        self.assertEqual([['name']], list(map(lambda x: x[0], auto.input_mapper.built_features)))

    def test_transform_no_response(self):
        data = pandas.DataFrame(data=['john clark', 'sue fox', 'mary lastname'], columns=['name'])

        # One variable
        text_vars = ['name']
        auto = Automater(text_vars=text_vars)
        auto.fit(data)

        (X, y) = auto.transform(data)

        # Find correct shape
        self.assertEqual((3, 2), X[0].shape)

        # Test output values
        self.assertEqual(None, y)

        # Test with unseen terms
        test_data = pandas.DataFrame(data=['Brendan Herger'], columns=['name'])
        (X_test, y_test) = auto.transform(test_data)
        self.assertTrue(numpy.array_equal([[0, 0]], X_test[0]))

        pass

    def test_create_input_nub(self):
        data = lib.load_titanic()

        # One variable
        text_vars = ['name']
        auto = Automater(text_vars=text_vars)
        auto.fit(data)

        self.assertEqual(1, len(auto.input_layers))

    def test_whole(self):
        data = lib.load_titanic()

        msk = numpy.random.rand(len(data)) < 0.95
        data_train = data[msk]
        data_test = data[~msk]

        text_vars = ['name']
        numerical_vars = ['survived']

        # Create auto
        auto = Automater(text_vars=text_vars, numerical_vars=numerical_vars, response_var='survived')

        # Train auto
        auto.fit(data_train)
        X_train, y_train = auto.transform(data)

        # Create model

        x = auto.input_nub
        x = Dense(30, activation='relu')(x)
        x = auto.output_nub(x)

        model = Model(inputs=auto.input_layers, outputs=x)
        model.compile(optimizer='Adam', loss=auto.loss)

        # Train DL model
        model.fit(X_train, y_train)

        # Transform test set
        data_test = data_test.drop('survived', axis=1)
        X_test, y_test = auto.transform(data_test)
        model.predict(X_test)

        pass
