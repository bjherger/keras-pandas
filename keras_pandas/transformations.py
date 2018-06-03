import logging
from collections import defaultdict

import numpy
from gensim.utils import simple_preprocess
from sklearn.base import BaseEstimator, TransformerMixin


class EmbeddingVectorizer(TransformerMixin, BaseEstimator):
    def __init__(self, embedding_sequence_length=None):
        # TODO Allow for UNK 'dropout' rate

        self.embedding_sequence_length = embedding_sequence_length

        # Create a dictionary, with default value 0 (corresponding to UNK token)
        self.token_index_lookup = defaultdict(int)
        self.token_index_lookup['UNK'] = 0
        self.next_token_index = 1

        pass

    def fit(self, X, y=None):
        # Format text for processing, by creating a list of strings
        observation_series = map(lambda x: x[0], X)

        # Preprocess & tokenize
        observation_series = map(simple_preprocess, observation_series)

        # Generate embedding_sequence_length, if necessary
        if self.embedding_sequence_length is None:
            self.embedding_sequence_length = self.generate_embedding_sequence_length(observation_series)

        # Update index_lookup
        tokens = [val for sublist in observation_series for val in sublist]
        logging.info('Fitting with tokens: {}'.format(tokens))

        for token in tokens:
            if token not in self.token_index_lookup:
                self.token_index_lookup[token] = self.next_token_index
                self.next_token_index += 1

        return self

    def transform(self, X):

        # TODO Preprocess & tokenize

        # TODO Convert tokens to indices

        # TODO Pad embedding length

        # TODO Format for outgoing
        # X = map(lambda x: [x], observation_series)
        # X = numpy.ndarray(X)
        pass

    def generate_embedding_sequence_length(self, observation_series):
        logging.info('Generating embedding_sequence_length')
        lengths = map(len, observation_series)
        embedding_sequence_length = int(numpy.median(lengths))
        logging.info('Generated embedding_sequence_length: {}'.format(embedding_sequence_length))

        return embedding_sequence_length
