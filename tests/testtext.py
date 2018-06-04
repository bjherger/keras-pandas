import logging
import unittest

import numpy
from keras.engine import Layer
from keras_pandas.Automater import Automater

from keras_pandas import lib

logging.getLogger().setLevel(logging.DEBUG)


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
        pass
