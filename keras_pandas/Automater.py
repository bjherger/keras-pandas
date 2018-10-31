import copy
import logging
from functools import reduce

from sklearn_pandas import DataFrameMapper

from keras_pandas.data_types.Numerical import Numerical


class Automater(object):

    def __init__(self, data_type_dict=dict(), output_var=None, data_type_handlers=dict()):

        self.datatype_variable_dict = data_type_dict

        self.variable_datatype_dict = dict()
        self.input_vars = copy.copy(reduce(lambda x, y: x + y, self.datatype_variable_dict.values()))

        # If there's an output_var, remove it from from input_vars
        if (output_var is not None) and (output_var in self.input_vars):
            self.input_vars.remove(output_var)

        self.output_var = output_var
        self.supervised = self.output_var is not None
        self.input_mapper = None
        self.output_mapper = None
        self.fitted = False

        self.datatype_handlers = {'numerical': Numerical()}
        self.datatype_handlers.update(data_type_handlers)

        for datatype_name, variable_list in self.datatype_variable_dict.items():
            for variable in variable_list:
                handler = self.datatype_handlers.get(datatype_name, None)
                self.variable_datatype_dict[variable] = handler

        # Exit checks
        self._valid_configurations_check()

    def fit(self, input_dataframe):
        # Setup checks
        self._check_input_df(input_dataframe)

        # Create mappers
        self.input_mapper = self._create_mapper(self.input_vars)
        self.output_mapper = self._create_mapper(self.output_var)

        # TODO Update fitted to True
        pass

    def transform(self, input_dataframe, df_out=None):
        # TODO Setup checks
        self._check_fitted()
        self._check_input_df(input_dataframe)

        # TODO Transform input variables
        # TODO Format data for return

        # TODO Check if response_var is in input dataframe
        # TODO Transform output variable
        # TODO Add response variable to return_df

        # TODO Check if df_out
        if df_out:
            # TODO Return correctly formatted DF
            pass
        else:
            # TODO Return correctly formatted Numpy objects
            pass

    def fit_transform(self, input_dataframe):
        """
        Perform a `fit`, and then a `transform`. See `transform` for return documentation

        """
        return self.fit(input_dataframe).transform(input_dataframe)

    def suggest_loss(self):
        self._check_fitted()
        self._check_has_response_var()
        # TODO Look up datatype class for respone variable
        # TODO Extract suggested loss from datatype class
        # TODO Return suggested loss
        pass

    def inverse_transform_output(self, y):
        self._check_fitted()
        self._check_has_response_var()
        pass

    def _create_input_nub(self):
        # TODO
        pass

    def _create_output_nub(self):
        self._check_has_response_var()
        pass

    def _create_mapper(self, variable_list):
        transformation_list = list()
        logging.info('Creating mapper for variables: {}'.format(variable_list))
        for variable in variable_list:
            datatype_name = self.variable_datatype_dict[variable]

            # Pull the default pipeline
            datatype = self.datatype_handlers[datatype_name]
            default_pipeline = datatype.default_transformation_pipeline

            # Copy the default pipeline, so each variable has its own learned parameters
            variable_pipeline = list(map(copy.deepcopy, default_pipeline))

            # Add to the aggregator
            transformation_list.append(([variable], variable_pipeline))

            logging.info('Creating transformation pipeline for variable: {}, '
                         'with datatype_name: {} and transformation_list: '
                         '{}'.format(variable, datatype_name, transformation_list))

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
                raise ValueError('Output variable: {} is not in variable_datatype_dict: {}. Please add output variable to '
                                 'data type dict.'.format(self.output_var, self.variable_datatype_dict))

            # Check that respone_var 's datatype class supports output
            if not output_datatype.supports_output:
                raise ValueError('Output variable: {} has been assigned datatype: {}. However, this datatype does not '
                                 'support being used as an output variable'.format(self.output_var, output_datatype))

        return True
