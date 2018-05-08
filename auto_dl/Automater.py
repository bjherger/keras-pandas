import copy
import logging

from keras.layers import Concatenate, Dense
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
        self._user_provided_variables = [item for sublist in self._variable_type_dict.values() for item in
                                         sublist]

        # Create transformation pipeline from defaults
        self.sklearn_mapper_pipelines = copy.deepcopy(constants.default_sklearn_mapper_pipelines)

        # Create mapper, to transform input variables
        self._sklearn_pandas_mapper = self._create_sklearn_pandas_mapper(self._variable_type_dict)

        # Create input variable type handler
        self.input_nub_type_handlers = constants.default_input_nub_type_handlers

        # Initialize Keras nubs
        self.input_layers = None
        self.input_nub = None
        self.output_nub = None

        # TODO
        self._datetime_expansion_method_dict = None

        # TODO
        self._embedding_size_function = None

        # TODO
        self._variable_transformer_dict = None

    def fit(self, input_dataframe, y=None):
        # TODO Validate input dataframe

        # Fit _sklearn_pandas_mapper with input dataframe
        # TODO Allow users to fit on dataframes that do not contain y variable
        logging.info('Fitting mapper w/ response_var: {}'.format(self.response_var))
        self._sklearn_pandas_mapper.fit(input_dataframe)

        # Transform input dataframe, for use to create Keras input layers
        self._sklearn_pandas_mapper.transform(input_dataframe)

        # Initialize & set input layers
        input_layers, input_nub = self._create_input_nub(self._variable_type_dict, input_dataframe)
        self.input_layers = input_layers
        self.input_nub = input_nub

        # TODO Initialize & set output layer(s)
        if y is not None:
            self.output_nub = self._create_output_nub(self._variable_type_dict, input_dataframe, y=y)

        # Set self.fitted to True
        self.fitted = True

        return self

    def transform(self, dataframe):

        # Reference var
        response_var_filled = False

        # Check for missing _user_provided_variables
        missing_vars = set(self._user_provided_variables).difference(dataframe.columns)

        # Check if response_var is set, and is listed in missing vars
        if self.response_var is not None and self.response_var in missing_vars:
            logging.warn('Filling response var: {} with None, for transformation'.format(self.response_var))
            missing_vars.remove(self.response_var)
            dataframe[self.response_var] = None
            response_var_filled = True

        # Check if any remaining _user_provided_variables are missing
        if len(missing_vars) > 0:
            raise ValueError('Provided dataframe is missing variables: {}'.format(missing_vars))

        # TODO Expand variables, as necessary

        # Transform dataframe w/ SKLearn-pandas
        transformed_df = self._sklearn_pandas_mapper.transform(dataframe)
        logging.info('Created transformed_df, w/ columns: {}'.format(list(transformed_df.columns)))

        # Remove 'response var', which was filled w/ None values
        if response_var_filled:
            logging.warn('Removing filled response var: {}'.format(self.response_var))
            transformed_df = transformed_df.drop(self.response_var, axis=1)

        if self.df_out:
            return transformed_df
        else:
            if self.response_var is not None and response_var_filled is False:
                X = transformed_df.drop(self.response_var, axis=1).as_matrix()
                y = transformed_df[self.response_var].tolist()
            else:
                X = transformed_df.as_matrix()
                y = None
            return X, y

    def fit_transform(self, dataframe):
        # TODO
        pass

    def set_embedding_size_function(self, embedding_size_function):
        # TODO
        pass

    def set_embedding_size(self, variable, embedding_size):
        # TODO
        pass

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

                if variable not in self._user_provided_variables:
                    raise ValueError(
                        'Unknown input variable: {}, which is not in list of input variables'.format(variable))
                elif variable not in input_dataframe.columns:
                    raise ValueError('Given variable: {} is not in transformed dataframe columns: {}'
                                     .format(variable, input_dataframe.columns))

                if variable == self.response_var and self.response_var is not None:
                    logging.info('Not creating an input layer for response variable: {}'.format(self.response_var))
                    continue

                # Apply handler to current variable, creating nub input and nub tip
                variable_input, variable_input_nub_tip = variable_type_handler(variable, input_dataframe)
                input_layers.append(variable_input)
                input_nub_tips.append(variable_input_nub_tip)

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

    def _create_output_nub(self, _variable_type_dict, input_dataframe, y):
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
        else:
            raise NotImplementedError(
                'Output layer for variable type: {} not yet implemented'.format(response_variable_type))

        return output_nub

        # TODO Create appropriate output layer

        # TODO Return output layer

    def _create_sklearn_pandas_mapper(self, _variable_type_dict):

        transformation_list = list()

        # Iterate through all variable types
        for (variable_type, variable_list) in _variable_type_dict.items():
            logging.info('Working variable type: {}, with variable list: {}'.format(variable_type, variable_list))

            # Extract default transformation pipeline
            default_pipeline = self.sklearn_mapper_pipelines[variable_type]
            logging.info('For variable type: {}, using default pipeline: {}'.format(variable_type, default_pipeline))

            for variable in variable_list:
                logging.debug('Creating transformation for variable: {}'.format(variable))

                transformation_list.append(([variable], default_pipeline))

        logging.info('Created transformation pipeline: {}'.format(transformation_list))
        mapper = DataFrameMapper(transformation_list, df_out=True)

        return mapper
