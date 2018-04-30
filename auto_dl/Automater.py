import copy
import logging

from keras.layers import Concatenate, Dense
from sklearn_pandas import DataFrameMapper

import lib
import constants


class Automater(object):
    def __init__(self, numerical_vars=list(), categorical_vars=list(), boolean_vars=list(), datetime_vars=list(),
                 non_transformed_vars=list(), df_out=False):

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

        # Create list of user provided input variables
        self._user_provided_input_variables = [item for sublist in self._variable_type_dict.values() for item in sublist]

        # TODO
        self._output_variables = None

        # Create transformation pipeline from defaults
        self.sklearn_mapper_pipelines = copy.deepcopy(constants.default_sklearn_mapper_pipelines)

        # Create mapper, to transform input variables
        self._sklearn_pandas_mapper = self._create_sklearn_pandas_mapper(self._variable_type_dict, self.df_out)

        # Create input variable type handler
        self.input_nub_type_handlers = constants.default_input_nub_type_handlers

        # TODO
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
        self._sklearn_pandas_mapper.fit(input_dataframe)

        # Transform input dataframe, for use to create input layers
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
        # TODO

        # TODO Check variables

        # TODO Expand variables, as necessary

        # TODO Transform expanded dataframe
        transformed = self._sklearn_pandas_mapper.transform(dataframe)

        # TODO Return transformed dataframe
        return transformed

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

                if variable not in self._user_provided_input_variables:
                    raise ValueError(
                        'Unknown input variable: {}, which is not in list of input variables'.format(variable))

                if variable not in input_dataframe.columns:
                    raise ValueError('Given variable: {} is not in transformed dataframe columns: {}'
                                     .format(variable, input_dataframe.columns))

                # Apply handler to current variable, creating nub input and nub tip
                variable_input, variable_input_nub_tip = variable_type_handler(variable, input_dataframe)
                input_layers.append(variable_input)
                input_nub_tips.append(variable_input_nub_tip)

        # TODO Concatenate nub tips
        logging.info('Creating input_nub, by concatenating input_nub_tips: {}'.format(input_nub_tips))
        input_nub = Concatenate(input_nub_tips, name='concatenate_inputs')

        return input_layers, input_nub

    def _create_output_nub(self, _variable_type_dict, input_dataframe, y):
        logging.info('Creating output nub, for variable: {}'.format(y))

        # Find which variable type for response variable
        response_variable_types = filter(lambda (key, value): y in value, _variable_type_dict.items())
        logging.info('Found response variable type(s)'.format(response_variable_types))
        if len(y) <1:
            raise ValueError('Response variable: {} is not in provided variable type lists'.format(y))
        elif len(y) > 1:
            raise ValueError('Response variable: {} appears in more than one provided variable type lists'.format(
                y))

        response_variable_type = response_variable_types[0]

        output_nub = None

        if response_variable_type == 'numerical_vars':
            # Create Dense layer w/ single node
            output_nub = Dense(units=1, activation='linear')
        else:
            raise NotImplementedError('Output layer for variable type: {} not yet implemented'.format(response_variable_type))

        return output_nub



        # TODO Create appropriate output layer

        # TODO Return output layer


    def _create_sklearn_pandas_mapper(self, _variable_type_dict, df_out):

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
        mapper = DataFrameMapper(transformation_list, df_out=df_out)

        return mapper
