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
        auto = Automater(numerical_vars=numerical_vars, timeseries_vars=timeseries_vars, response_var='ise')

        # Fit automater
        auto.fit(train_observations)

        # Create model
        x = auto.input_nub
        x = Dense(32)(x)
        x = auto.output_nub(x)

        model = Model(inputs=auto.input_layers, outputs=x)
        model.compile(optimizer='adam', loss=auto.loss)

        # Train model
        train_X, train_y = auto.transform(train_observations)
        print(len(train_X))
        print(train_X[0].shape)
        model.fit(train_X, train_y)


        # TODO Use model to predict
        pass


