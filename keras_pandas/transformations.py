import logging
from collections import defaultdict

import numpy
from gensim.utils import simple_preprocess
from sklearn.base import BaseEstimator, TransformerMixin


class EmbeddingVectorizer(TransformerMixin, BaseEstimator):
    """
    Converts text into padded sequences. The output of this transformation is consistent with the required format
    for Keras embedding layers

    For example `'the fat man`' might be transformed into `[2, 0, 27, 1, 1, 1]`, if the `embedding_sequence_length` is
    6.

    There are a few sentinel values used by this layer:

     - `0` is used for the UNK token (tokens which were not seen during training)
     - `1` is used for the padding token (to fill out sequences that shorter than `embedding_sequence_length`)

    """

    def __init__(self, embedding_sequence_length=None):
        # TODO Allow for UNK 'dropout' rate

        self.embedding_sequence_length = embedding_sequence_length

        # Create a dictionary, with default value 0 (corresponding to UNK token)
        self.token_index_lookup = defaultdict(int)
        self.token_index_lookup['UNK'] = 0
        self.token_index_lookup['__PAD__'] = 1
        self.next_token_index = 2

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

        # Undo Numpy formatting
        observations = map(lambda x: x[0], X)

        # Convert to embedding format
        observations = map(self.process_string, observations)

        # Redo numpy formatting
        observations = map(lambda x: numpy.array(x), observations)

        return numpy.matrix(observations)


    def generate_embedding_sequence_length(self, observation_series):
        logging.info('Generating embedding_sequence_length')
        lengths = map(len, observation_series)
        embedding_sequence_length = int(numpy.median(lengths))
        logging.info('Generated embedding_sequence_length: {}'.format(embedding_sequence_length))

        return embedding_sequence_length

    def process_string(self, input_string):
        """
        Turn a string into padded sequences, consisten with Keras's Embedding layer

         - Simple preprocess & tokenize
         - Convert tokens to indices
         - Pad sequence to be the correct length

        :param input_string: A string, to be converted into a padded sequence of token indices
        :type input_string: str
        :return: A padded, fixed-length array of token indices
        :rtype: [int]
        """
        logging.debug('Processing string: {}'.format(input_string))

        # Convert to tokens
        tokens = simple_preprocess(input_string)
        logging.debug('Tokens: {}'.format(tokens))

        # Convert to indices
        indices = map(lambda x: self.token_index_lookup[x], tokens)
        logging.debug('Indices: {}'.format(indices))

        # Pad indices
        padding_index = self.token_index_lookup['__PAD__']
        padding_length = self.embedding_sequence_length
        padded_indices = self.pad(indices, length=padding_length, pad_char=padding_index)
        logging.debug('Padded indices: {}'.format(padded_indices))

        return padded_indices

    @staticmethod
    def pad(input_sequence, length, pad_char):
        """
        Pad the given iterable, so that it is the correct length.

        :param input_sequence: Any iterable object
        :param length: The desired length of the output.
        :type length: int
        :param pad_char: The character or int to be added to short sequences
        :type pad_char: str or int
        :return: A sequence, of len `length`
        :rtype: []
        """

        # If input_sequence is a string, convert to to an explicit list
        if isinstance(input_sequence, str):
            input_sequence = list(input_sequence)

        # If the input_sequence is the correct length, return it
        if len(input_sequence) == length:
            return input_sequence

        # If the input_sequence is too long, truncate it
        elif len(input_sequence) > length:
            return input_sequence[:length]

        # If the input_sequence is too short, extend it w/ the pad_car
        else:
            padding_len = length - len(input_sequence)
            padding = [pad_char] * padding_len
            return input_sequence + padding