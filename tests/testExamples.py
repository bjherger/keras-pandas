from examples import lending_club_predict_loan_status
from tests.testbase import TestBase


class TestExamples(TestBase):

    def test_lending_club_predict_loan_status(self):
        lending_club_predict_loan_status.main()
