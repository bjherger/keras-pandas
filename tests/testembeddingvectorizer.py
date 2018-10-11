import logging
import unittest

from numpy.testing import assert_array_equal
from sklearn_pandas import DataFrameMapper

from keras_pandas import lib
from keras_pandas.transformations import EmbeddingVectorizer
from tests.testbase import TestBase


class TestEmbeddingVectorizer(TestBase):

    def test_padding(self):
        # Empty string
        data = ''
        padding_len = 10
        padded_data = EmbeddingVectorizer.pad(data, padding_len, 's')
        self.assertEqual(padding_len, len(padded_data))
        self.assertEqual(['s', 's', 's', 's', 's', 's', 's', 's', 's', 's'], padded_data)

        # Real string
        data = 'attack of the'
        padding_len = 20
        padded_data = EmbeddingVectorizer.pad(data, padding_len, '*')
        self.assertEqual(padding_len, len(padded_data))
        self.assertEqual(list('attack of the*******'), padded_data)

        # Indices
        data = range(5)
        padding_len = 20
        padded_data = EmbeddingVectorizer.pad(data, padding_len, 19)
        self.assertEqual(padding_len, len(padded_data))
        self.assertEqual([0, 1, 2, 3, 4, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19], padded_data)

        # Large number of indices
        data = range(2, 5)
        padding_len = 220
        padded_data = EmbeddingVectorizer.pad(data, padding_len, 0)
        self.assertEqual(padding_len, len(padded_data))
        self.assertEqual(list(range(2, 5)) + [0] * 217, padded_data)

    def test_empty_strings(self):
        data = lib.load_titanic()
        data = data[['name']]

        ev = EmbeddingVectorizer()

        ev.fit(data)

    def test_mapper(self):
        data = lib.load_titanic()

        transformation_list = [(['name'], [EmbeddingVectorizer(embedding_sequence_length=12)])]

        mapper = DataFrameMapper(transformation_list, df_out=True)

        mapper.fit(data)

        data_transformed = mapper.transform(data)

        assert_array_equal([2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 1, 1], data_transformed.values[0,:])
