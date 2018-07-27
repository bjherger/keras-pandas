from examples import mushrooms, titanic
from tests.testbase import TestBase


class TestExamples(TestBase):

    def test_mushrooms(self):
        mushrooms.main()

    def test_titanic(self):
        titanic.main()
