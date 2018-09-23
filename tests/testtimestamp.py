import logging

from keras_pandas.Automater import Automater
from keras_pandas import lib
from tests.testbase import TestBase

logging.getLogger().setLevel(logging.INFO)

class TestText(TestBase):


    def test_fit(self):
        data = lib.load_lending_club()

        timestamp_vars = ['issue_d', 'earliest_cr_line']

        auto = Automater(timestamp_vars=timestamp_vars)
        auto.fit(data)

        self.assertTrue(auto.fitted)
        self.assertEqual(2, len(auto.input_layers))

    def test_transform(self):
        data = lib.load_lending_club()

        timestamp_vars = ['issue_d', 'earliest_cr_line']

        auto = Automater(timestamp_vars=timestamp_vars)
        auto.fit(data)

        transformed = auto.transform(data)

        pass

    def test_whole(self):
        data = lib.load_air_quality()

        timestamp_vars = ['issue_d', 'earliest_cr_line']

        auto = Automater(timestamp_vars=timestamp_vars)
        auto.fit(data)

        pass

