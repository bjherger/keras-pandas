import logging
import sys

from keras import Model
from keras.layers import Dense

from keras_pandas.Automater import Automater
from keras_pandas import lib
from tests.testbase import TestBase

logging.getLogger().setLevel(logging.INFO)

class TestText(TestBase):


    # def test_fit(self):
    #     data = lib.load_lending_club()
    #
    #     timestamp_vars = ['issue_d', 'earliest_cr_line']
    #
    #     auto = Automater(timestamp_vars=timestamp_vars)
    #     auto.fit(data)
    #
    #     self.assertTrue(auto.fitted)
    #     self.assertEqual(2, len(auto.input_layers))
    #
    # def test_transform(self):
    #     data = lib.load_lending_club()
    #
    #     timestamp_vars = ['issue_d', 'earliest_cr_line']
    #
    #     auto = Automater(timestamp_vars=timestamp_vars)
    #     auto.fit(data)
    #
    #     transformed = auto.transform(data)
    #
    #     pass

    def test_whole(self):
        data = lib.load_air_quality()

        timestamp_vars = ['date', 'time']
        numerical_vars = ['no2gt']
        print(data)

        auto = Automater(timestamp_vars=timestamp_vars, numerical_vars=numerical_vars, response_var='no2gt')
        auto.fit(data)

        X_train, y_train = auto.transform(data)

        # Extract input_nub from auto
        input_nub = auto.input_nub

        # Extract output_nub from auto
        output_nub = auto.output_nub

        # Create DL model
        x = input_nub
        # x = Dense(30)(x)
        x = output_nub(x)

        model = Model(inputs=auto.input_layers, outputs=x)
        model.compile(optimizer='Adam', loss='mse')

        # Train DL model
        print(X_train)
        print(y_train)
        model.fit(X_train, y_train, epochs=10)

        # Transform test set
        X_test, y_test = auto.transform(data)
        model.predict(X_test)

        pass

