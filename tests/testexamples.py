from examples import mushrooms, titanic, lending_club_classification, lending_club_regression
from tests.testbase import TestBase


class TestExamples(TestBase):

    def test_mushrooms(self):
        mushrooms.main()

    def test_titanic(self):
        titanic.main()

    def test_lending(self):
        lending_club_classification.main()

    def test_lending_regression(self):
        lending_club_regression.main()
