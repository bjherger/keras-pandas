import logging
from collections import defaultdict

import numpy
from gensim.utils import simple_preprocess
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


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
        observations = self.prepare_input(X)

        # Preprocess & tokenize
        observations = list(map(lambda x: simple_preprocess(x), observations))

        # Generate embedding_sequence_length, if necessary
        if self.embedding_sequence_length is None:
            self.embedding_sequence_length = self.generate_embedding_sequence_length(observations)

        # Update index_lookup
        tokens = [val for sublist in observations for val in sublist]
        logging.debug('Fitting with tokens: {}'.format(tokens))

        for token in tokens:
            if token not in self.token_index_lookup:
                self.token_index_lookup[token] = self.next_token_index
                self.next_token_index += 1

        return self

    def transform(self, X):

        observations = self.prepare_input(X)

        # Convert to embedding format
        observations = list(map(self.process_string, observations))

        # Redo numpy formatting
        observations = list(map(lambda x: numpy.array(x), observations))

        return numpy.matrix(observations)


    def generate_embedding_sequence_length(self, observation_series):
        lengths = list(map(len, observation_series))
        embedding_sequence_length = max([int(numpy.median(lengths)), 1])
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
        indices = list(map(lambda x: self.token_index_lookup[x], tokens))
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
            return list(input_sequence) + list(padding)

    @staticmethod
    def prepare_input(X):
        # Undo Numpy formatting
        observations = list(map(lambda x: x[0], X))

        observations = map(str, observations)
        return observations

import pandas as pd
import numpy as np








class CategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values from a categorical/string np.ndarray or pd.Series
    with the most frequent value on the training data.
    Parameters
    ----------
    missing_values : string or "NaN", optional (default="NaN")
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. None and np.nan are treated
        as being the same, use the string value "NaN" for them.
    copy : boolean, optional (default=True)
        If True, a copy of X will be created.
    strategy : string, optional (default = 'most_frequent')
        The imputation strategy.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.
    fill_value : string, optional (default='?')
        The value that all instances of `missing_values` are replaced
        with if `strategy` is set to `constant`. This is useful if
        you don't want to impute with the mode, or if there are multiple
        modes in your data and you want to choose a particular one. If
        `strategy` is not set to `constant`, this parameter is ignored.
    Attributes
    ----------
    fill_ : str
        The imputation fill value
    """

    def __init__(
        self,
        missing_values='NaN',
        strategy='most_frequent',
        fill_value='?',
        fill_unknown_labels=False,
        copy=True
    ):
        self.missing_values = missing_values
        self.copy = copy
        self.fill_value = fill_value
        self.strategy = strategy
        self.known_values = None
        self.fill_unknown_labels = fill_unknown_labels

        strategies = ['constant', 'most_frequent']
        if self.strategy not in strategies:
            raise ValueError(
                'Strategy {0} not in {1}'.format(self.strategy, strategies)
            )

    def fit(self, X, y=None):
        """
        Get the most frequent value.
        Parameters
        ----------
            X : np.ndarray or pd.Series
                Training data.
            y : Passthrough for ``Pipeline`` compatibility.
        Returns
        -------
            self: CategoricalImputer
        """

        mask = self._get_null_mask(X, self.missing_values)
        X = X[~mask]
        if self.strategy == 'most_frequent':
            modes = pd.Series(X).mode()
        elif self.strategy == 'constant':
            modes = np.array([self.fill_value])
        if modes.shape[0] == 0:
            raise ValueError('Data is empty or all values are null')
        elif modes.shape[0] > 1:
            raise ValueError('No value is repeated more than '
                             'once in the column')
        else:
            self.fill_ = modes[0]

        self.known_values = set(X)

        return self

    def transform(self, X):
        """
        Replaces missing values in the input data with the most frequent value
        of the training data.
        Parameters
        ----------
            X : np.ndarray or pd.Series
                Data with values to be imputed.
        Returns
        -------
            np.ndarray
                Data with imputed values.
        """

        check_is_fitted(self, 'fill_')

        if self.copy:
            X = X.copy()

        null_mask = self._get_null_mask(X, self.missing_values)
        X[null_mask] = self.fill_

        if self.fill_unknown_labels:
            unknown_label_mask = self._get_unknown_label_mask(X)
            X[unknown_label_mask] = self.fill_

        return np.asarray(X)

    @staticmethod
    def _get_null_mask(X, value):
        """
        Compute the boolean mask X == missing_values.
        """
        if value == "NaN" or \
                value is None or \
                (isinstance(value, float) and np.isnan(value)):
            return pd.isnull(X)
        else:
            return X == value

    def _get_unknown_label_mask(self, X):
        """
        Compute the boolean mask X == missing_values.
        """
        return numpy.logical_not(numpy.isin(X, self.known_values))