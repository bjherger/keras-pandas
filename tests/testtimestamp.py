import logging

from build.lib.keras_pandas.Automater import Automater
from keras_pandas import lib
from tests.testbase import TestBase

logging.getLogger().setLevel(logging.INFO)

class TestText(TestBase):

    def test_create_input_nub(self):
        data = lib.load_lending_club()

        timestamp_vars = ['issue_d']

        auto = Automater(timestamp_vars=timestamp_vars)
        auto.fit(data)

    def test_fit(self):
        pass

    def test_transform(self):
        pass

    def test_whole(self):
        pass

