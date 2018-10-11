from examples import numerical_lending_club
from tests.testbase import TestBase


class TestAutomater(TestBase):
    def test_numerical_lending_club(self):
        numerical_lending_club.main()