from keras_pandas.transformations import EpochTransformer
from tests.testbase import TestBase


class TestEpochTransformer(TestBase):

    def test_init(self):
        et = EpochTransformer()
        pass

    def test_fit(self):
        et = EpochTransformer()

        data = [['2018-09-12'], ['1970-01-01', '1776-07-04']]
        et = et.fit(data)
        self.assertEqual(EpochTransformer, type(et))
        pass

    def test_transform(self):
        et = EpochTransformer()

        data = [['2018-09-12'], ['1970-01-01'], ['1776-07-04']]
        transformed = et.fit_transform(data)
        
        self.assertEqual(0, transformed[0, 1])
        self.assertEqual(1536710400000000000, transformed[0, 0])
        self.assertEqual((1, 3), transformed.shape)
