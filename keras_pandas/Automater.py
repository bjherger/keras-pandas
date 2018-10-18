import copy
import logging
import sklearn

import numpy
import pandas
from keras.engine import Layer
from keras.layers import Concatenate, Dense
from sklearn_pandas import DataFrameMapper

from keras_pandas import constants, lib


class Automater(object):

    def __init__(self, numerical_vars=list(), categorical_vars=list(), boolean_vars=list(), datetime_vars=list(),
                 text_vars=list(), non_transformed_vars=list(), response_var=None, df_out=False):

        self.response_var = response_var
        self.fitted = False
        self.df_out = df_out

        # Set up variable type dict, with entries <variable_type, list of variables>
        self._variable_type_dict = dict()
        self._variable_type_dict['numerical_vars'] = numerical_vars
        # Categorical variables include both categorical and boolean
        self._variable_type_dict['categorical_vars'] = categorical_vars + boolean_vars
        self._variable_type_dict['datetime_vars'] = datetime_vars
        self._variable_type_dict['text_vars'] = text_vars
        self._variable_type_dict['non_transformed_vars'] = non_transformed_vars
        lib.check_variable_list_are_valid(self._variable_type_dict)

        # Create list of user provided input variables, by flattening values from _variable_type_dict
        self._user_provided_variables = [item for sublist in self._variable_type_dict.values() for item in sublist]
        if response_var is not None and response_var not in self._user_provided_variables:
            raise ValueError('Response variable: {} not in list of user provided variables: '
                             '{}'.format(response_var, self._user_provided_variables))

        # Create mappers, to transform input variables
        (self.input_mapper, self.output_mapper) = self._create_mappers(self._variable_type_dict)

        # Create input variable type handler
        self.input_nub_type_handlers = constants.default_input_nub_type_handlers

        # Initialize list of variables fed into Keras nubs
        self.keras_input_variable_list = list()

        # Initialize Keras nubs & layers
        self.input_layers = None
        self.input_nub = None
        self.output_nub = None

        # Initialize suggested Keras loss
        self.loss = None


    def fit(self, input_dataframe):
        """
        Get the data and layers ready for use

         - Train the input transformation pipelines
         - Create the keras input layers
         - Train the output transformation pipeline(s) (optional, only if there is a response variable)
         - Create the output layer(s) (optional, only if there is a response variable)
         - Set `self.fitted` to `True`

        :param input_dataframe:
        :return: self, now in a fitted state. The Automater now has initialized input layers, output layer(s) (if
            response variable is present), and can be used for the transform step
        :rtype: Automater
        """
        # TODO Validate input dataframe

        # Fit input_mapper with input dataframe
        logging.info('Fitting input mapper')
        self.input_mapper.fit(input_dataframe)

        # Transform input dataframe, for use to create Keras input layers
        input_variables_df = self.input_mapper.transform(input_dataframe)

        if self.response_var is not None:
            logging.info('Fitting response var: {}'.format(self.response_var))
            # Fit output mapper
            self.output_mapper.fit(input_dataframe)

            # Transform output data
            output_variables_df = self.output_mapper.transform(input_dataframe)

        # Initialize & set input layers
        # TODO Only create nubs if they do not exist yet (?)
        input_layers, input_nub = self._create_input_nub(self._variable_type_dict, input_variables_df)
        self.input_layers = input_layers
        self.input_nub = input_nub

        # Initialize & set output layer(s)
        if self.response_var is not None:
            # TODO Only create output nub if it doesn't exist yet (?)
            self.output_nub = self._create_output_nub(self._variable_type_dict, output_variables_df=output_variables_df,
                                                      y=self.response_var)

        # Initialize & set suggested loss
        if self.response_var is not None:
            self.loss = self._suggest_loss(self._variable_type_dict, y=self.response_var)

        # Set self.fitted to True
        self.fitted = True

        return self

    def transform(self, input_dataframe, df_out=None):
        """

         - Validate that the provided `input_dataframe` contains the required input columns
         - Transform the keras input columns
         - Transform the response variable, if it is present
         - Format the data for return

        :param input_dataframe: A pandas dataframe, containing all keras input layers
        :type input_dataframe: pandas.DataFrame
        :return: Either a pandas dataframe (if `df_out = True`), or a numpy object (if `df_out = False`). This object
            will contain: the transformed input variables, and the transformed output variables (if the output variable
            is present in `input_dataframe`

        """

        # Check if fitted yet
        if not self.fitted:
            raise ValueError('Cannot transform without being fitted first. Call fit() method before transform() method')

        # Check df_out state
        if df_out is None:
            df_out = self.df_out

        # Check if we have a response variable, and if it is available
        if self.response_var is not None and self.response_var in input_dataframe.columns:
            y_available = True
        else:
            y_available = False

        # Check if any input variables are missing
        missing_input_vars = set(self._user_provided_variables).difference(input_dataframe.columns)

        # Check if response_var is set, and is listed in missing vars
        if self.response_var is not None and y_available is False:
            logging.info('Response variable is set, but unavailable in df to be transformed. Not transforming response '
                         'variable')
            missing_input_vars.remove(self.response_var)

        # Check if any remaining _user_provided_variables are missing
        if len(missing_input_vars) > 0:
            raise ValueError('Provided dataframe is missing variables: {}'.format(missing_input_vars))

        # TODO Expand variables, as necessary

        # Create input variables df
        input_variables = self.input_mapper.transform(input_dataframe)
        logging.info('Created input_variables, w/ columns: {}'.format(list(input_variables.columns)))

        # Create output variables df
        if y_available:
            output_variables = self.output_mapper.transform(input_dataframe)
            logging.info('Created output_variables, w/ columns: {}'.format(list(output_variables.columns)))

        if df_out:
            # Join input and output dfs on index
            if y_available:
                df_out = input_variables.join(output_variables)
            else:
                df_out = input_variables
            return df_out
        else:

            X = list()
            for variable in self.keras_input_variable_list:
                logging.info('Adding keras input variable: {} to X'.format(variable))
                if variable in input_variables.columns:
                    data = input_variables[variable].values
                else:
                    logging.info('Checking for derived variables')
                    variable_name_prefix = variable + '_'
                    derived_variable_list = list(filter(lambda x: x.startswith(variable_name_prefix),
                                                   input_variables.columns))
                    logging.debug('Derived variable list: {}'.format(derived_variable_list))
                    data = input_variables[derived_variable_list].values
                X.append(data)
            if y_available:
                y = output_variables[self.response_var].values
            else:
                y = None
            return X, y

    def fit_transform(self, input_dataframe):
        """
        Perform a `fit`, and then a `transform`. See `transform` for return documentation

        """
        return self.fit(input_dataframe).transform(input_dataframe)

    def _get_variable_type(self, variable_type_dict, variable):
        pass

    def _create_input_nub(self, variable_type_dict, input_dataframe):
        """

        Generate a 'nub', appropriate for use as an input (and possibly additional Keras layers). Each Keras input
        variable has on input pipeline, with:

         - One  Input (required)
         - Possible additional layers (optional, such as embedding layers for text)

        All input pipelines are then joined with a Concatenate layer

        :param variable_type_dict: A dictionary, with keys describing variables types, and values listing particular
            variables
        :type variable_type_dict: {str:[str]}
        :param input_dataframe: A pandas dataframe, containing all keras input layers
        :type input_dataframe: pandas.DataFrame
        :return: A Keras layer, which can be fed into future layers
        :rtype: ([keras,Input], Layer)
        """

        logging.info('Beginning creation of input nubs and input nub tips for _variable_type_dict: {}'.format(
            variable_type_dict))

        # Set up reference variables

        # Input layers
        input_layers = list()

        # Input nub tips (nub tip = the last layer for a specific input. This is the layer that is connected to the rest
        # of the network)
        input_nub_tips = list()

        # Iterate through variable types
        # TODO Iterate through handled variable types, rather than given variable types. Ordering could matter.
        for (variable_type, variable_list) in variable_type_dict.items():
            logging.info('Creating input nubs for variable_type: {}'.format(variable_type))

            if len(variable_list) <= 0:
                logging.info('Variable type {} has 0 corresponding variables. Skipping.'.format(variable_type))
                continue

            # Pull correct handler for variable type
            if variable_type in self.input_nub_type_handlers:
                variable_type_handler = self.input_nub_type_handlers[variable_type]
            else:
                raise ValueError('No handler for provided variable_type: {}'.format(variable_type))

            # Iterate through variables for current variable type
            for variable in variable_list:
                logging.debug('Creating input nub for variable type: {}, variable: {}'.format(variable_type, variable))

                if variable == self.response_var and self.response_var is not None:
                    logging.info('Not creating an input layer for response variable: {}'.format(self.response_var))
                    continue
                elif variable not in self._user_provided_variables:
                    raise ValueError(
                        'Unknown input variable: {}, which is not in list of input variables'.format(variable))
                elif variable not in input_dataframe.columns:

                    # Check for derived variable (e.g. `name` is turned into `name_0` and `name_1`
                    variable_name_prefix = variable + '_'
                    derived_variable_list = list(filter(lambda x: x.startswith(variable_name_prefix),
                                                   input_dataframe.columns))
                    if len(derived_variable_list) <= 0:
                        raise ValueError('Given variable: {} is not in transformed dataframe columns: {}'
                                         .format(variable, input_dataframe.columns))

                # Apply handler to current variable, creating nub input and nub tip
                logging.info('Creating inputs for variable: {}, of variable type: {}'.format(variable, variable_type))
                variable_input, variable_input_nub_tip = variable_type_handler(variable, input_dataframe)
                input_layers.append(variable_input)
                input_nub_tips.append(variable_input_nub_tip)
                self.keras_input_variable_list.append(variable)

        # Concatenate nub tips
        if len(input_nub_tips) > 1:
            logging.info('Creating input_nub, by concatenating input_nub_tips: {}'.format(input_nub_tips))
            input_nub = Concatenate(name='concatenate_inputs')(input_nub_tips)
        elif len(input_nub_tips) == 1:
            logging.info('Only one variable input: {}. Return that input nub, instead of concatenating'.format(
                input_nub_tips[0]))
            input_nub = input_nub_tips[0]
        else:
            logging.warn('No inputs provided for model. Returning None for input nub.')
            input_nub = None

        return input_layers, input_nub

    def _create_output_nub(self, variable_type_dict, output_variables_df, y):
        """
        Generate a 'nub', appropriate for use as an output / final Keras layer.

        The structure of this nub will depend on the y variable's data type

        :param variable_type_dict: A dictionary, with keys describing variables types, and values listing particular
            variables
        :type variable_type_dict: {str:[str]}
        :param output_variables_df: A dataframe containing the output variable. This is necessary for some data types
            (e.g. a categorical output needs to know how levels the categorical variable has)
        :type output_variables_df: pandas.DataFrame
        :param y: The name of the response variable
        :type y: str
        :return: A single Keras layer, correctly formatted to output the response variable provided
        :rtype: Layer
        """
        logging.info('Creating output nub, for variable: {}'.format(y))

        # Find response variable's variable type
        response_variable_types = lib.get_variable_type(y, variable_type_dict, self.response_var)
        response_variable_type = response_variable_types[0]
        logging.info('Found response variable type'.format(response_variable_type))


        if response_variable_type == 'numerical_vars':
            # Create Dense layer w/ single node
            output_nub = Dense(units=1, activation='linear')

        elif response_variable_type == 'categorical_vars':
            # +1 for UNK level
            categorical_num_response_levels = len(set(output_variables_df[self.response_var])) + 1
            output_nub = Dense(units=categorical_num_response_levels, activation='softmax')
        else:
            raise NotImplementedError(
                'Output layer for variable type: {} not yet implemented'.format(response_variable_type))

        return output_nub

    def _create_mappers(self, variable_type_dict):
        """
        Creates two sklearn-pandas mappers, one for the input variables, and another for the output variable(s)

        :param variable_type_dict: A dictionary, with keys describing variables types, and values listing particular
            variables
        :type variable_type_dict: {str:[str]}
        :return: Two sklearn-pandas mappers, one for the input variables, and another for the output variable(s)
        :rtype: (DataFrameMapper, DataFrameMapper)
        """

        sklearn_mapper_pipelines = constants.default_sklearn_mapper_pipelines
        input_transformation_list = list()
        output_transformation_list = list()

        # Iterate through all variable types
        for (variable_type, variable_list) in variable_type_dict.items():
            logging.info('Working variable type: {}, with variable list: {}'.format(variable_type, variable_list))

            # Extract default transformation pipeline
            default_pipeline = sklearn_mapper_pipelines[variable_type]
            logging.info('For variable type: {}, using default pipeline: {}'.format(variable_type, default_pipeline))

            for variable in variable_list:

                variable_pipeline = list(map(copy.deepcopy, default_pipeline))
                logging.info('Creating transformation for variable: {}, '
                              'with pipeline: {}'.format(variable, variable_pipeline))

                # Append to the correct list
                if variable == self.response_var:
                    logging.debug('Response var: {} is being added to output mapper'.format(variable))

                    output_transformation_list.append(([variable], variable_pipeline))
                else:
                    logging.debug('Input var: {} is being added  to input mapper'.format(variable))
                    input_transformation_list.append(([variable], variable_pipeline))

        logging.info('Creating input transformation pipeline: {}'.format(input_transformation_list))
        logging.info('Creating output transformation pipeline: {}'.format(output_transformation_list))
        input_mapper = DataFrameMapper(input_transformation_list, df_out=True)
        output_mapper = DataFrameMapper(output_transformation_list, df_out=True)

        return input_mapper, output_mapper

    def _suggest_loss(self, variable_type_dict, y):
        # Find response variable's variable type
        logging.info('Finding suggested loss, for variable: {}'.format(y))

        # Find response variable's variable type
        response_variable_types = lib.get_variable_type(y, variable_type_dict, self.response_var)
        response_variable_type = response_variable_types[0]
        logging.info('Found response variable type'.format(response_variable_type))

        # Look up suggested loss
        if response_variable_type in constants.default_suggested_losses:
            suggested_loss = constants.default_suggested_losses[response_variable_type]
            logging.info('Suggesting loss: {}'.format(suggested_loss))
        else:
            raise ValueError('No default loss for variable {}, '
                             'with variable_type: {}'.format(y, response_variable_type))

        return suggested_loss

    def inverse_transform_output(self, y):
        """

        :param y:
        :return:
        """
        # Find response variable's variable type
        response_variable_types = lib.get_variable_type(self.response_var, self._variable_type_dict, self.response_var)
        response_variable_type = response_variable_types[0]
        logging.info('Found response variable type: {}'.format(response_variable_type))

        # Get transformation pipeline for response variable
        response_transform_tuple = list(filter(lambda x: x[0][0] == self.response_var, self.output_mapper.built_features))[0]
        response_transform_pipeline = response_transform_tuple[1]
        logging.info('response_transform_pipeline" {}'.format(response_transform_pipeline))

        # Parse and inverse transform y based on response variable type
        if response_variable_type is 'numerical_vars':
            response_variable_transformer = response_transform_pipeline.named_steps['standardscaler']
            logging.info('StandardScaler was trained for response_var, and is being used for inverse transform. '
                         'scale_: {}, mean_: {}, var_: {}'.
                         format(response_variable_transformer.scale_, response_variable_transformer.mean_,
                                response_variable_transformer.var_))
        elif response_variable_type is 'categorical_vars':
            response_variable_transformer = response_transform_pipeline.named_steps['labelencoder']
            logging.info('LabelEncoder was trained for response_var, and is being used for inverse transform. '
                         'classes_: {}'.format(
                response_variable_transformer.classes_))

            # Find the index of the most likely response
            y = numpy.argmax(y, axis=1)
        else:
            raise ValueError('Unable to perform inverse transform for response variable data type: {}'.format(
                response_variable_type))

        natural_scaled_vars = response_variable_transformer.inverse_transform(y)
        return natural_scaled_vars

