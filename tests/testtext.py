import logging
import unittest

import numpy
from keras import losses, Model
from keras.layers import Dense

from keras_pandas import lib
from keras_pandas.Automater import Automater

logging.getLogger().setLevel(logging.INFO)


class TestText(unittest.TestCase):

    def test_fit(self):
        data = lib.load_titanic()
        # One variable
        text_vars = ['name']

        auto = Automater(text_vars=text_vars)
        auto.fit(data)

        self.assertEqual(Automater, type(auto))
        self.assertEqual(text_vars, auto._user_provided_variables)
        self.assertTrue(auto.fitted)

        self.assertEqual([['name']], map(lambda x: x[0], auto.input_mapper.built_features))

    def test_transform_no_response(self):
        data = lib.load_titanic()

        # One variable
        text_vars = ['name']
        auto = Automater(text_vars=text_vars)
        auto.fit(data)

        (X, y) = auto.transform(data)

        # TODO Find correct shape
        self.assertEqual((887, 4), X[0].shape)
        self.assertTrue(numpy.array_equal([2, 3, 4, 5], X[0][0]))
        self.assertEqual(None, y)

        # TODO Test output values
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
        categorical_vars = ['survived']

        # Create auto
        auto = Automater(text_vars=text_vars, categorical_vars=categorical_vars, response_var='survived')

        # Train auto
        auto.fit(data_train)
        X_train, y_train = auto.transform(data)

        # Create model

        x = auto.input_nub
        x = Dense(30, activation='relu')(x)
        x = auto.output_nub(x)

        model = Model(inputs=auto.input_layers, outputs=x)
        model.compile(optimizer='Adam', loss=losses.sparse_categorical_crossentropy)

        # Train DL model
        model.fit(X_train, y_train)

        # Transform test set
        data_test = data_test.drop('survived', axis=1)
        X_test, y_test = auto.transform(data_test)
        model.predict(X_test)

        pass
