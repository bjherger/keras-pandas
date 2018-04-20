import logging

from sklearn_pandas import DataFrameMapper

import lib
import constants


class Automater(object):
    def __init__(self, numerical_vars=list(), categorical_vars=list(), boolean_vars=list(), datetime_vars=list(),
                 non_transformed_vars=list()):
        # Set up variable type dict, with entries <variable_type, list of variables>
        self._variable_type_dict = dict()
        self._variable_type_dict['numerical_vars'] = numerical_vars
        self._variable_type_dict['categorical_vars'] = categorical_vars
        self._variable_type_dict['boolean_vars'] = boolean_vars
        self._variable_type_dict['datetime_vars'] = datetime_vars
        self._variable_type_dict['non_transformed_vars'] = non_transformed_vars
        lib.check_variable_list_are_valid(self._variable_type_dict)

        # Create mapper, to transform input variables
        self._sklearn_pandas_mapper = self._create_sklearn_pandas_mapper(self._variable_type_dict)

        # TODO
        self._datetime_expansion_method_dict = None

        # TODO
        self._embedding_size_function = None

        # TODO
        self._variable_transformer_dict = None

        # TODO
        self._input_variables = [item for sublist in self._variable_type_dict.values() for item in sublist]

        # TODO
        self._output_variables = None

        # TODO
        self._input_nub = None

        # TODO
        self._output_nub = None

    def fit(self, dataframe):
        # TODO
        pass

    def transform(self, dataframe):
        # TODO
        pass

    def fit_transform(self, dataframe):
        # TODO
        pass

    def get_input_nub(self):
        # TODO
        pass

    def get_output_nub(self):
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

    def list_input_variables(self):
        # TODO
        pass

    def list_output_variables(self):
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

    @staticmethod
    def _create_sklearn_pandas_mapper(_variable_type_dict):

        transformation_list = list()

        # Iterate through all variable types
        for (variable_type, variable_list) in _variable_type_dict.items():
            logging.info('Working variable type: {}, with variable list: {}'.format(variable_type, variable_list))

            # Extract default transformation pipeline
            default_pipeline = constants.mapper_default_pipelines[variable_type]
            logging.info('For variable type: {}, using default pipeline: {}'.format(variable_type, default_pipeline))

            for variable in variable_list:
                logging.debug('Creating transformation for variable: {}'.format(variable))

                transformation_list.append((variable, default_pipeline))

        logging.info('Created transformation pipeline: {}'.format(transformation_list))
        mapper = DataFrameMapper(transformation_list)

        return mapper
