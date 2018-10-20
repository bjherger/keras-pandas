import numpy

import pandas
import unittest

from keras import Model, losses
from keras.layers import Dense
from sklearn.model_selection import train_test_split

from keras_pandas import lib
from keras_pandas.Automater import Automater
from tests.testbase import TestBase


class TestTimeSeries(TestBase):

    def test_transform_no_response(self):
        pass

    def test_timeseries_whole(self):
        observations = lib.load_instanbul_stocks(as_ts=True)

        # Train test split
        train_observations, test_observations = train_test_split(observations)
        train_observations = train_observations.copy()
        test_observations = test_observations.copy()

        # Create data type lists
        timeseries_vars = ['ise_lagged', 'sp_lagged']
        numerical_vars = ['ise']


        # Create automater
        auto = Automater(numerical_vars=numerical_vars, timeseries_vars=timeseries_vars)

        # Fit automater
        auto.fit(train_observations)

        # TODO Create model
        x = auto.input_nub
        x = Dense(32)(x)
        x = auto.output_nub(x)

        Model(inputs=auto.input_nub, outputs=auto.output_nub)


        # TODO Train model

        # TODO Use model to predict
        pass


