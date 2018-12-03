import logging

import keras
import numpy
from keras import losses
from keras.layers import Embedding, Flatten, Dense

from keras_pandas import lib
from keras_pandas.transformations import StringEncoder, CategoricalImputer, LabelEncoder


class Categorical():
    def __init__(self):
        self.supports_output = True
        self.default_transformation_pipeline = [StringEncoder(),
                                                CategoricalImputer(strategy='constant', fill_value='UNK',
                                                                   fill_unknown_labels=True),
                                                LabelEncoder()]

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
        # Get transformed data for shaping
        transformed = transformed_observations[variable].as_matrix()

        # Set up dimensions for input_layer layer
        if len(transformed.shape) >= 2:
            input_sequence_length = int(transformed.shape[1])
        else:
            input_sequence_length = 1

        # TODO Convert below to numpy.max (?)
        categorical_num_levels = int(max(transformed)) + 2
        embedding_output_dim = int(min((categorical_num_levels + 1) / 2, 50))

        logging.info('Creating embedding for cat_var: {}, with input_sequence_length: {}, categorical_num_levels: {}, '
                     'and embedding_output_dim: {}'.format(variable, input_sequence_length, categorical_num_levels,
                                                           embedding_output_dim))

        input_layer = keras.Input(shape=(input_sequence_length,),
                                  name=lib.namespace_conversion('input_{}'.format(variable)))
        x = input_layer
        x = Embedding(input_dim=categorical_num_levels,
                      output_dim=embedding_output_dim,
                      input_length=input_sequence_length,
                      name=lib.namespace_conversion('embedding_{}'.format(variable)))(x)
        x = Flatten(name=lib.namespace_conversion('flatten_embedding_{}'.format(variable)))(x)

        input_nub = x

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
        # +1 for UNK level
        categorical_num_response_levels = len(set(transformed_observations[variable])) + 1
        output_layer = Dense(units=categorical_num_response_levels, activation='softmax')

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
        response_variable_transformer = response_transform_pipeline.named_steps['labelencoder']
        logging.info('LabelEncoder was trained for response_var, and is being used for inverse transform. '
                     'classes_: {}'.format(response_variable_transformer.classes_))

        # Find the index of the most likely response
        natural_scaled_vars = numpy.argmax(y_pred, axis=1)

        return natural_scaled_vars

    def output_suggested_loss(self):
        self._check_output_support()
        suggested_loss = losses.sparse_categorical_crossentropy
        return suggested_loss

    def _check_output_support(self):
        if not self.supports_output:
            raise ValueError('This datatype: {} does not support output, but has called to an output related '
                             'function.'.format(self.__class__))
        return True
