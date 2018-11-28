import copy
import logging
import pandas
from functools import reduce

from keras.layers import Concatenate
from sklearn_pandas import DataFrameMapper

from keras_pandas.data_types.Categorical import Categorical
from keras_pandas.data_types.Numerical import Numerical
from keras_pandas.data_types.Text import Text
from keras_pandas.data_types.TimeSeries import TimeSeries


class Automater(object):

    def __init__(self, data_type_dict=dict(), output_var=None, datatype_handlers=dict()):
        """
        An Automater object, allows users to rapidly build and iterate on deep learning models.

        This class supports building and iterating on deep learning models by providing:

         - A cleaned, transformed and correctly formatted X and y (good for keras, sklearn or any other ML platform)
         - An `input_nub`, without the hassle of worrying about input shapes or data types
         - An `nub`, correctly formatted for the kind of response variable provided

        :param data_type_dict: A dictionary, in the format {'datatype': ['variable_name_1', 'variable_name_2']}
        :type data_type_dict: {str:[str]}
        :param output_var: The name of the response variable
        :type output_var: str
        :param datatype_handlers: Any custom or external datatype handlers, in the format {'datatype': DataTypeClass}
        :type datatype_handlers: {str:class}
        """

        # Dictionary of the format {'datatype': ['variable_name_1', 'variable_name_2']}
        self.datatype_variable_dict = data_type_dict

        # Set up a list of all input variables
        self.input_vars = copy.copy(reduce(lambda x, y: x + y, self.datatype_variable_dict.values()))

        # If there's an output_var, remove it from from input_vars
        if (output_var is not None) and (output_var in self.input_vars):
            self.input_vars.remove(output_var)

        # Set up output
        self.output_var = output_var
        self.supervised = self.output_var is not None

        # Set up datatype handlers
        self.datatype_handlers = {'numerical': Numerical(),
                                  'categorical': Categorical(),
                                  'boolean': Categorical(),
                                  'timeseries': TimeSeries(),
                                  'text': Text()}

        # Add user-supplied datatype handlers
        self.datatype_handlers.update(datatype_handlers)

        # Dictionary of the format {'variable_name_1': DataTypeClass}
        self.variable_datatype_dict = dict()
        for datatype_name, variable_list in self.datatype_variable_dict.items():
            for variable in variable_list:
                if datatype_name in self.datatype_handlers:
                    handler = self.datatype_handlers.get(datatype_name, None)
                    self.variable_datatype_dict[variable] = handler
                    logging.info('Providing variable: {} with datatype handler: {}'.format(variable, handler))
                else:
                    raise ValueError('Unknown datatype: {}'.format(datatype_name))

        # Set up mappers
        self.input_mapper = self._create_mapper(self.input_vars)
        if self.supervised:
            self.output_mapper = self._create_mapper([self.output_var])
        else:
            self.output_mapper = None

        # Attributes
        self.fitted = False
        self.input_layers = None
        self.input_nub = None
        self.output_nub = None

        # Exit checks
        self._valid_configurations_check()

    def fit(self, observations):
        """

         - Fit input mapper
         - Create input layer and nub
         - Create output mapper (if supervised)
         - Create output nub (if supervised)
        - Set `self.fitted` to `True`

        :param observations: A pandas DataFrame, containing the relevant variables
        :type observations: pandas.DataFrame
        :return: self, now in a fitted state. The Automater now has initialized input layers, output layer(s) (if
            response variable is present), and can be used for the transform step
        :rtype: Automater
        """
        # Setup checks
        self._check_input_df(observations)

        # Fit input mapper, and transform data for layer creation
        input_observations_transformed = self.input_mapper.fit_transform(observations)

        # Create input layer and nub

        input_layers, input_nub = self._create_input_nub(input_observations_transformed)
        self.input_layers = input_layers
        self.input_nub = input_nub

        if self.supervised:
            # Fit output mapper, and transform data for layer creation
            output_transformed_dataframe = self.output_mapper.fit_transform(observations, )

            # Create output nub
            self.output_nub = self._create_output_nub(output_transformed_dataframe)

        # Update fitted to True
        self.fitted = True
        return self

    def transform(self, observations, df_out=False):
        """
         - Transform the keras input columns
         - Transform the output_var, if supervised and the output_var is present
         - Format the data, consistent w/ df_out

        :param observations: A pandas dataframe, containing all keras input layers
        :type observations: pandas.DataFrame
        :param df_out: Whether to return a Pandas DataFrame. Returns DataFrame if True, keras-compatable object if
        false
        :type df_out: bool
        :return: Either a pandas dataframe (if `df_out = True`), or a numpy object (if `df_out = False`). This object
            will contain: the transformed input variables, and the transformed output variables (if the output variable
            is present in `input_dataframe`
        """

        # Setup checks
        self._check_fitted()
        self._check_input_df(observations)

        # Transform input variables
        input_observations_transformed = self.input_mapper.transform(observations)

        # Transform output_var if supervised and available
        if self.supervised and self.output_var in observations:
            # Transform output variable
            output_observations_transformed = self.output_mapper.transform(observations)
        else:
            output_observations_transformed = None

        # Format data and return
        if df_out:
            # Return correctly formatted DF
            if output_observations_transformed is not None:
                output = pandas.concat([input_observations_transformed, output_observations_transformed], axis=1)
            else:
                output = input_observations_transformed
            return output
        else:
            # Return correctly formatted Numpy objects as X, y

            # Format X as a list of arrays, consistent w/ Keras's input formatting
            X = list()
            for variable in self.input_vars:
                logging.info('Adding keras input variable: {} to X'.format(variable))
                if variable in input_observations_transformed.columns:
                    data = input_observations_transformed[variable].values
                else:
                    logging.info('Checking for derived variables')
                    variable_name_prefix = variable + '_'
                    derived_variable_list = list(filter(lambda x: x.startswith(variable_name_prefix),
                                                        input_observations_transformed.columns))
                    logging.debug('Derived variable list: {}'.format(derived_variable_list))
                    data = input_observations_transformed[derived_variable_list].values
                X.append(data)

            if output_observations_transformed is not None:
                return X, output_observations_transformed[self.output_var].values
            else:
                return X, None

    def fit_transform(self, observations):
        """
        Perform a `fit`, and then a `transform`. See `transform` for return documentation

        """
        return self.fit(observations).transform(observations)

    def suggest_loss(self):
        """
        Suggest a loss function, based on:

         - Output variable datatype
         - Observations of output_var

        :return: A Keras supported loss function
        """

        self._check_fitted()
        self._check_has_response_var()

        # Look up datatype class for respone variable
        datatype = self.variable_datatype_dict[self.output_var]

        # Extract suggested loss from datatype class
        suggested_loss = datatype.output_suggested_loss()

        # Return suggested loss
        return suggested_loss

    def inverse_transform_output(self, y):
        """
        Transform the output_var to be in the same basis (scale / domain) as it was in the original data set. This is
        convenient for comparing predictions to actual data, and computing metrics relative to actual data and other
        models
        :param y: The output of a Keras model's .predict function
        :type y: numpy.ndarray
        :return: Data, which can be compared to the original data set
        :rtype numpy.ndarray
        """
        self._check_fitted()
        self._check_has_response_var()

        # Look up datatype class for respone variable
        datatype = self.variable_datatype_dict[self.output_var]

        # Pull fitted response_transform_pipeline
        response_transform_tuple = \
        list(filter(lambda x: x[0][0] == self.output_var, self.output_mapper.built_features))[0]
        response_transform_pipeline = response_transform_tuple[1]

        # Use data type to output_inverse_transform variable
        raw_scaled_output = datatype.output_inverse_transform(y, response_transform_pipeline)
        return raw_scaled_output

        pass

    def _create_input_nub(self, transformed_observations):
        """
        Generate a nub, appropriate for feeding all input variables into a Keras model. Each input variable has one
        input layer and one input nub, with:

          - One  Input (required)
          - Possible additional layers (optional, such as embedding layers for text)

        All input nubs are then joined with a Concatenate layer
        :param transformed_observations: A pandas dataframe, containing all keras input variables after their
        transformation pipelines
        :type transformed_observations: pandas.DataFrame
        :return: A Keras layer, which can be fed into future layers
        :rtype: ([keras,Input], Layer)
        """
        # Initialize input_layer_list
        input_layer_list = list()

        # Initialize input_nub_list
        input_nub_list = list()

        # Iterate through input variables and datatypes
        for variable in self.input_vars:
            datatype = self.variable_datatype_dict[variable]
            logging.info('Creating input_layer and input_nub for variable: {} and datatype: {}'.format(variable,
                                                                                                       datatype))

            # Use datatype to create input and nub
            input_layer, input_nub = datatype.input_nub_generator(variable, transformed_observations)

            # Add input to input_layer_list
            input_layer_list.append(input_layer)

            # Add input_nub to input_nub_list
            input_nub_list.append(input_nub)

        # Concatenate input_nubs, if there is more than one
        if len(input_nub_list) > 1:
            logging.info('Creating input_nub, by concatenating input_nub_list: {}'.format(input_nub_list))
            input_nub = Concatenate(name='concatenate_inputs')(input_nub_list)
        elif len(input_nub_list) == 1:
            logging.info('Only one variable input: {}. Return that input nub, instead of concatenating'.format(
                input_nub_list[0]))
            input_nub = input_nub_list[0]
        else:
            logging.warning('No inputs provided for model. Returning None for input nub.')
            input_nub = None

        return input_layer_list, input_nub

    def _create_output_nub(self, output_observations_transformed):
        self._check_has_response_var()

        # Pull datatype for output_var
        datatype = self.variable_datatype_dict[self.output_var]

        # Check that datatype supports output
        if not datatype.supports_output:
            raise ValueError('datatype: {} does not support output, but is used as the datatype for the '
                             'output_var'.format(datatype))

        # Use datatype to create output_nub
        output_nub = datatype.output_nub_generator(self.output_var, output_observations_transformed)

        return output_nub

    def _create_mapper(self, variable_list):
        transformation_list = list()
        logging.info('Creating mapper for variables: {}'.format(variable_list))
        for variable in variable_list:
            # Pull the default pipeline
            datatype = self.variable_datatype_dict[variable]
            default_pipeline = datatype.default_transformation_pipeline

            # Copy the default pipeline, so each variable has its own learned parameters
            variable_pipeline = list(map(copy.deepcopy, default_pipeline))

            # Add to the aggregator
            transformation_list.append(([variable], variable_pipeline))

            logging.info('Creating transformation pipeline for variable: {}, '
                         'with datatype: {} and transformation_list: '
                         '{}'.format(variable, type(datatype), transformation_list))

        logging.info('Creating transformation pipeline: {}'.format(transformation_list))

        mapper = DataFrameMapper(transformation_list, df_out=True)
        return mapper

    def _check_fitted(self):
        if not self.fitted:
            raise AssertionError('Automater has not been fitted yet. Please call to Automater.fit() with appropriate '
                                 'data to fit the model. ')
        else:
            return True

    def _check_has_response_var(self):
        if self.output_var is None:
            raise AssertionError('Attempting to call to function that requires a response variable. Please create a new'
                                 'automater, using the response_var parater')
        else:
            return True

    def _check_input_df(self, input_dataframe):
        # TODO Check that input_dataframe contains all variables, except for response variable
        pass

    def _valid_configurations_check(self):
        # Check that each variable is assigned to only one variable type
        for outer_datatype, outer_variable_list in self.datatype_variable_dict.items():
            for inner_datatype, inner_variable_list in self.datatype_variable_dict.items():

                # Do not compare data types to themselves
                if inner_datatype == outer_datatype:
                    continue

                else:
                    intersection = set(outer_variable_list).intersection(set(inner_variable_list))
                    if len(intersection) > 0:
                        raise ValueError('Datatype lists {} and {} overlap, and share variables(s): {}'.
                                         format(inner_datatype, outer_datatype, intersection))

        # Check that all datatype handlers are available
        if not set(self.datatype_variable_dict.keys()).issubset(self.datatype_handlers):
            difference = set(self.datatype_variable_dict.keys()).difference(self.datatype_handlers)
            raise ValueError('No handler for datatype(s): {}'.format(difference))

        if self.supervised:
            # Extract output datatype
            output_datatype = self.variable_datatype_dict.get(self.output_var, None)

            # Check that response variable is in the data_type_dict
            if output_datatype is None:
                raise ValueError(
                    'Output variable: {} is not in variable_datatype_dict: {}. Please add output variable to '
                    'data type dict.'.format(self.output_var, self.variable_datatype_dict))

            # Check that respone_var 's datatype class supports output
            if not output_datatype.supports_output:
                raise ValueError('Output variable: {} has been assigned datatype: {}. However, this datatype does not '
                                 'support being used as an output variable'.format(self.output_var, output_datatype))

        return True
