import logging

import keras
import numpy
from keras.layers import Bidirectional, LSTM, Embedding

from keras_pandas import lib
from keras_pandas.transformations import StringEncoder, EmbeddingVectorizer


class Text():
    def __init__(self):
        self.supports_output = False
        self.default_transformation_pipeline = [StringEncoder(), EmbeddingVectorizer()]

    @staticmethod
    def input_nub_generator(variable, transformed_observations):
        """
        Generate an input layer and input 'nub' for a Keras network.

         - input_layer: The input layer accepts data from the outside world.
         - input_nub: The input nub will always include the input_layer as its first layer. It may also include
         other layers for handling the data type in specific ways

        :param variable: Name of the variable
        :type variable: str
        :param transformed_observations: A dataframe, containing either the specified variable, or derived variables
        :type transformed_observations: pandas.DataFrame
        :return: A tuple containing the input layer, and the last layer of the nub
        """
        logging.info('Creating input nub for: {}'.format(variable))
        # Get transformed data for shaping. One column per token.
        if variable in transformed_observations.columns:
            variable_list = [variable]
        else:
            variable_name_prefix = variable + '_'
            variable_list = list(filter(lambda x: x.startswith(variable_name_prefix), transformed_observations.columns))
        logging.info('Determined variable list: {}'.format(variable_list))

        # Pull transformed data as matrix
        transformed = transformed_observations[variable_list].as_matrix()

        # Determine sequence length
        if len(transformed.shape) >= 2:
            # If we have multiple columns, it's one column per word
            input_sequence_length = int(transformed.shape[1])
        else:
            # If there are not multiple columns, there is only one word
            input_sequence_length = 1

        # Determine vocabulary size (number of rows in the embedding). The additional offsets are due to 1  for len
        # vs indexing w/ 0, 1 for unknown token, and the others for something else?
        vocab_size = int(numpy.max(transformed)) + 4

        # Determine embedding output size
        # TODO There must be a better heuristic
        embedding_output_dim = 200

        logging.info('Creating embedding for text_var: {}, with input_sequence_length: {}, vocab size: {}, '
                     'and embedding_output_dim: {}'.format(variable, input_sequence_length, vocab_size,
                                                           embedding_output_dim))

        # Create and stack layers
        input_layer = keras.Input(shape=(input_sequence_length,),
                                  name=lib.namespace_conversion('input_{}'.format(variable)))
        x = input_layer
        x = Embedding(input_dim=vocab_size, output_dim=embedding_output_dim, input_length=input_sequence_length,
                      name=lib.namespace_conversion('embedding_{}'.format(variable)))(x)
        x = Bidirectional(LSTM(128,
                               name=lib.namespace_conversion('lstm_{}'.format(variable))),
                          name=lib.namespace_conversion('bidirectiona_lstm_{}'.format(variable)))(x)

        input_nub = x

        # Return
        return input_layer, input_nub

    def output_nub_generator(self, variable, transformed_observations):
        """
        Generate an output layer for a Keras network.

         - output_layer: A keras layer, which is formatted to correctly accept the response variable

        :param variable: A Variable contained in the input_df
        :type variable: str
        :param transformed_observations: A dataframe, containing either the specified variable, or derived variables
        :type transformed_observations: pandas.DataFrame
        :return: output_layer
        """
        self._check_output_support()
        output_layer = None

        return output_layer

    def output_inverse_transform(self, y_pred, response_transform_pipeline):
        """
        Undo the transforming that was done to get data into a keras model. This inverse transformation will
        render the observations so they can be compared to the data in the natural scale provided by the user
        :param response_transform_pipeline: An SKLearn transformation pipeline, trained on the same variable as the
        model which produced y_pred
        :param y_pred: The data predicted by keras
        :return: The same data, in the natural basis
        """
        self._check_output_support()
        natural_scaled_vars = None

        return natural_scaled_vars

    def output_suggested_loss(self):
        self._check_output_support()
        suggested_loss = None
        return suggested_loss

    def _check_output_support(self):
        if not self.supports_output:

            raise ValueError('This datatype: {} does not support output, but has called to an output related '
                             'function.'.format(self.__class__))
        return True
