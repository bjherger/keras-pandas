import numpy

import pandas
import unittest

from keras import Model, losses
from keras.layers import Dense

from keras_pandas import lib
from keras_pandas.Automater import Automater
from tests.testbase import TestBase


class TestTimeSeries(TestBase):

    def test_transform_no_response(self):
        pass

    def test_categorical_whole(self):
        observations = lib.load_instanbul_stocks(as_ts=True)

        # TODO Train test split

        # TODO Create data type lists

        # TODO Create automater

        # TODO Create model

        # TODO Train model

        # TODO Use model to predict
        pass


