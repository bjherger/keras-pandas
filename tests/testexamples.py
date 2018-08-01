from examples import mushrooms, titanic, lending_club
from tests.testbase import TestBase


class TestExamples(TestBase):

    def test_mushrooms(self):
        mushrooms.main()

    def test_titanic(self):
        titanic.main()

    def test_lending(self):
        lending_club.main()
