from examples import numerical_response_lending_club, categorical_response_lending_club, categorical_response_mushrooms, categorical_response_titanic
from tests.testbase import TestBase
import numpy


class TestExamples(TestBase):
    def test_numerical_lending_club(self):
        numpy.random.seed(0)
        numerical_response_lending_club.main()

    def test_categorical_lending_club(self):
        numpy.random.seed(0)
        categorical_response_lending_club.main()

    def test_categorical_mushrooms(self):
        numpy.random.seed(0)
        categorical_response_mushrooms.main()

    def test_categorical_titanic(self):
        numpy.random.seed(0)
        categorical_response_titanic.main()

