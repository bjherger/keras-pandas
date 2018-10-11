from examples import numerical_lending_club
from tests.testbase import TestBase
import numpy


class TestExamples(TestBase):
    def test_numerical_lending_club(self):
        numpy.random.seed(0)
        numerical_lending_club.main()
