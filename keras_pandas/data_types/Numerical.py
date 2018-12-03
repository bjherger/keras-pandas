import logging

import keras
from keras.layers import Dense
from sklearn.preprocessing import Imputer, StandardScaler

from keras_pandas import lib


class Numerical():
    def __init__(self):
        self.supports_output = True
        self.default_transformation_pipeline = [Imputer(strategy='mean'), StandardScaler()]

    @staticmethod
    def input_nub_generator(variable, transformed_observations):
        """
        Generate an input layer and input 'nub' for a keras network.

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

        # Create input_layer layer
        input_layer = keras.Input(shape=(input_sequence_length,), dtype='float32',
                                  name=lib.namespace_conversion('input_{}'.format(variable)))
        input_nub = input_layer

        # Return, in format of input_layer, last variable-specific layer
        return input_layer, input_nub

    def output_nub_generator(self, variable, transformed_observations):
        """
                Generate an output layer for a Keras network.

                 - output_layer: A keras layer, which is formatted to correctly accept the response variable

                :param variable: A Variable contained in the input_df
                :type variable: str
                :param transformed_observations: A dataframe, containing either the specified variable, or derived
                variables
                :type transformed_observations: pandas.DataFrame
                :return: output_layer
                """
        self._check_output_support()
        output_nub = Dense(units=1, activation='linear')
        return output_nub

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

        # Get correct step
        response_variable_transformer = response_transform_pipeline.named_steps['standardscaler']
        logging.info('StandardScaler was trained for response_var, and is being used for inverse transform. '
                     'scale_: {}, mean_: {}, var_: {}'.
                     format(response_variable_transformer.scale_, response_variable_transformer.mean_,
                            response_variable_transformer.var_))

        natural_scaled_vars = response_variable_transformer.inverse_transform(y_pred)

        return natural_scaled_vars

    def output_suggested_loss(self):
        self._check_output_support()
        # TODO We can do better than this, if we are able to look at the response data
        suggested_loss = keras.losses.mean_squared_error
        return suggested_loss

    def _check_output_support(self):
        if not self.supports_output:

            raise ValueError('This datatype: {} does not support output, but has called to an output related '
                             'function.'.format(self.__class__))
