import logging
import unittest

from sklearn_pandas import DataFrameMapper

from keras_pandas import lib
from keras_pandas.transformations import EmbeddingVectorizer

logging.getLogger().setLevel(logging.INFO)
class TestEmbeddingVectorizer(unittest.TestCase):

    def test_empty_strings(self):
        data = lib.load_titanic()
        data = data[['name']]

        ev = EmbeddingVectorizer()

        ev.fit(data)

    def test_pipeline(self):
        data = lib.load_titanic()

        transformation_list = [(['name'], [EmbeddingVectorizer()])]

        mapper = DataFrameMapper(transformation_list, df_out=True)

        mapper.fit(data)

        data_transformed = mapper.transform(data)
