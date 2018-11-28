class AbstractDatatype():
    def __init__(self):
        self.supports_output = False
        self.default_transformation_pipeline = []

    def input_nub_generator(self, variable, transformed_obervations):
        """
        Generate an input layer and input 'nub' for a Keras network.

         - input_layer: The input layer accepts data from the outside world.
         - input_nub: The input nub will always include the input_layer as its first layer. It may also include
         other layers for handling the data type in specific ways

        :param variable: Name of the variable
        :type variable: str
        :param transformed_obervations: A dataframe, containing either the specified variable, or derived variables
        :type transformed_obervations: pandas.DataFrame
        :return: A tuple containing the input layer, and the last layer of the nub
        """

        input_layer = None
        input_nub = None

        return input_layer, input_nub

    def output_nub_generator(self, variable, input_observations):
        """
        Generate an output layer for a Keras network.

         - output_layer: A keras layer, which is formatted to correctly accept the response variable

        :param variable: A Variable contained in the input_df
        :type variable: str
        :param input_observations: A dataframe, containing either the specified variable, or derived variables
        :type input_observations: pandas.DataFrame
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
