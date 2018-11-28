from keras import Model
from sklearn_pandas import DataFrameMapper

from examples import lending_club_predict_loan_status
from keras_pandas import lib
from keras_pandas.data_types.Categorical import Categorical
from keras_pandas.data_types.Numerical import Numerical
from tests.testbase import TestBase


class TestExamples(TestBase):

    def test_lending_club_predict_loan_status(self):
        lending_club_predict_loan_status.main()