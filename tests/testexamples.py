from examples import numerical_lending_club, categorical_lending_club, categorical_mushrooms, categorical_titanic
from tests.testbase import TestBase
import numpy


class TestExamples(TestBase):
    def test_numerical_lending_club(self):
        numpy.random.seed(0)
        numerical_lending_club.main()

    def test_categorical_lending_club(self):
        numpy.random.seed(0)
        categorical_lending_club.main()

    def test_categorical_mushrooms(self):
        numpy.random.seed(0)
        categorical_mushrooms.main()

    def test_categorical_titanic(self):
        numpy.random.seed(0)
        categorical_titanic.main()

