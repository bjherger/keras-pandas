import copy
import logging
import pandas

import numpy
from keras.layers import Concatenate, Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import column_or_1d
from sklearn_pandas import DataFrameMapper

import constants
import lib


class Automater(object):

    def __init__(self, numerical_vars=list(), categorical_vars=list(), boolean_vars=list(), datetime_vars=list(),
                 non_transformed_vars=list(), response_var=None, df_out=False):

        self.response_var = response_var
        self.fitted = False
        self.df_out = df_out

        # Set up variable type dict, with entries <variable_type, list of variables>
        self._variable_type_dict = dict()
        self._variable_type_dict['numerical_vars'] = numerical_vars
        self._variable_type_dict['categorical_vars'] = categorical_vars
        self._variable_type_dict['boolean_vars'] = boolean_vars
        self._variable_type_dict['datetime_vars'] = datetime_vars
        self._variable_type_dict['non_transformed_vars'] = non_transformed_vars
        lib.check_variable_list_are_valid(self._variable_type_dict)

        # Create list of user provided input variables, by flattening values from _variable_type_dict
        self._user_provided_variables = [item for sublist in self._variable_type_dict.values() for item in sublist]

        # Create mappers, to transform input variables
        (self.input_mapper, self.output_mapper) = self._create_mappers(self._variable_type_dict)

        # Create input variable type handler
        self.input_nub_type_handlers = constants.default_input_nub_type_handlers

        # Initialize Keras nubs
        self.input_layers = None
        self.input_nub = None
        self.output_nub = None

        # Initialize list of variables fed into Keras nubs
        self.keras_input_variable_list = list()

        # TODO
        self._datetime_expansion_method_dict = None

    def fit(self, input_dataframe):
        # TODO Validate input dataframe

        # Fit input_mapper with input dataframe
        logging.info('Fitting input mapper')
        self.input_mapper.fit(input_dataframe)

        # Transform input dataframe, for use to create Keras input layers
        input_variables_df = self.input_mapper.transform(input_dataframe)

        if self.response_var is not None:
            # Fit output mapper

            self.output_mapper.fit(input_dataframe)

            # Transform output data
            output_variables_df = self.output_mapper.transform(input_dataframe)

        # Initialize & set input layers
        input_layers, input_nub = self._create_input_nub(self._variable_type_dict, input_variables_df)
        self.input_layers = input_layers
        self.input_nub = input_nub

        # Initialize & set output layer(s)
        if self.response_var is not None:
            # TODO Update to refer to correct method signature
            self.output_nub = self._create_output_nub(self._variable_type_dict, output_variables_df=output_variables_df, y=self.response_var)

        # Set self.fitted to True
        self.fitted = True

        return self

    def transform(self, dataframe):

        # Check if fitted yet
        if not self.fitted:
            raise ValueError('Cannot transform without being fitted first. Call fit() method before transform() method')

        # Check if we have a response variable, and if it is available
        if self.response_var is not None and self.response_var in dataframe.columns:
            y_available = True
        else:
            y_available = False

        # Check if any input variables are missing
        missing_input_vars = set(self._user_provided_variables).difference(dataframe.columns)

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
        input_variables = self.input_mapper.transform(dataframe)
        logging.info('Created input_variables, w/ columns: {}'.format(list(input_variables.columns)))

        # Create output variables df
        if y_available:
            output_variables = self.output_mapper.transform(dataframe)
            logging.info('Created output_variables, w/ columns: {}'.format(list(output_variables.columns)))

        if self.df_out:
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
                data = input_variables[variable].values
                X.append(data)
            if y_available:
                y = output_variables[self.response_var].values
            else:
                y = None
            return X, y

    def fit_transform(self, dataframe):
        return self.fit(dataframe).transform(dataframe)

    def get_transformers(self):
        # TODO
        pass

    def get_transformer(self, variable):
        # TODO
        pass

    def list_default_transformation_pipelines(self):
        # TODO
        pass

    def _check_input_dataframe_columns_(self, input_dataframe):
        # TODO
        pass

    def _check_output_dataframe_columns_(self, output_dataframe):
        # TODO
        pass

    def _datetime_expansion_(self, dataframe):
        # TODO
        pass

    def _create_input_nub(self, _variable_type_dict, input_dataframe):

        logging.info('Beginning creation of input nubs and input nub tips for _variable_type_dict: {}'.format(
            _variable_type_dict))

        # Set up reference variables

        # Input layers
        input_layers = list()

        # Input nub tips (nub tip = the last layer for a specific input. This is the layer that is connected to the rest
        # of the network)
        input_nub_tips = list()

        # Iterate through variable types
        # TODO Iterate through handled variable types, rather than given variable types. Ordering could matter.
        for (variable_type, variable_list) in _variable_type_dict.items():
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
                    raise ValueError('Given variable: {} is not in transformed dataframe columns: {}'
                                     .format(variable, input_dataframe.columns))

                # Apply handler to current variable, creating nub input and nub tip
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

    def _create_output_nub(self, _variable_type_dict, output_variables_df, y):
        logging.info('Creating output nub, for variable: {}'.format(y))

        # Find which variable type for response variable
        response_variable_types = filter(lambda (key, value): y in value, _variable_type_dict.items())
        response_variable_types = map(lambda (key, value): key, response_variable_types)
        logging.info('Found response variable type(s)'.format(response_variable_types))
        if len(response_variable_types) < 1:
            raise ValueError('Response variable: {} is not in provided variable type lists'.format(y))
        elif len(response_variable_types) > 1:
            raise ValueError('Response variable: {} appears in more than one provided variable type lists'.format(
                y))

        response_variable_type = response_variable_types[0]

        if response_variable_type == 'numerical_vars':
            # Create Dense layer w/ single node
            output_nub = Dense(units=1, activation='linear')

        elif response_variable_type == 'categorical_vars':
            categorical_num_response_levels = len(set(output_variables_df[self.response_var]))
            output_nub = Dense(units=categorical_num_response_levels, activation='softmax')
        else:
            raise NotImplementedError(
                'Output layer for variable type: {} not yet implemented'.format(response_variable_type))

        return output_nub


    def _create_mappers(self, _variable_type_dict):

        sklearn_mapper_pipelines = constants.default_sklearn_mapper_pipelines
        input_transformation_list = list()
        output_transformation_list = list()

        # Iterate through all variable types
        for (variable_type, variable_list) in _variable_type_dict.items():
            logging.info('Working variable type: {}, with variable list: {}'.format(variable_type, variable_list))

            # Extract default transformation pipeline
            default_pipeline = sklearn_mapper_pipelines[variable_type]
            logging.info('For variable type: {}, using default pipeline: {}'.format(variable_type, default_pipeline))

            for variable in variable_list:
                logging.debug('Creating transformation for variable: {}, '
                              'with default_pipeline: {}'.format(variable, default_pipeline))
                variable_pipeline = map(copy.copy, default_pipeline)

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
