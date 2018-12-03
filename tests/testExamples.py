from examples import lending_club_predict_loan_status, lending_club_predict_dti, instanbul_predict_ise, \
    titianic_predict_survived
from tests.testbase import TestBase


class TestExamples(TestBase):

    def test_lending_club_predict_loan_status(self):
        lending_club_predict_loan_status.main()

    def test_lending_club_predict_dti(self):
        lending_club_predict_dti.main()

    def test_instanbul_predict_ise(self):
        instanbul_predict_ise.main()

    def test_titianic_predict_survived(self):
        titianic_predict_survived.main()
