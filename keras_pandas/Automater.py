from functools import reduce


class Automater(object):

    def __init__(self, data_type_dict=dict(), output_var=None, data_type_handlers=dict()):

        self.data_type_dict = data_type_dict
        self.input_vars = reduce(lambda x, y: x + y, self.data_type_dict.values())

        # If there's an output_var, remove it from from input_vars
        if (output_var is not None) and (output_var in self.input_vars):
            self.input_vars.remove(output_var)

        self.output_var = output_var
        self.supervised = self.output_var != None
        self.input_mapper = None
        self.output_mapper = None
        self.fitted = False

        self.data_type_handlers = {'numerical': None,
                                   'categorical': None}

        # Exit checks
        self._valid_configurations_check()

    def fit(self, input_dataframe):
        # TODO Setup checks
        self._check_input_df(input_dataframe)

        # TODO Create mappers

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
        self._check_response_var()
        # TODO Look up datatype class for respone variable
        # TODO Extract suggested loss from datatype class
        # TODO Return suggested loss
        pass

    def inverse_transform_output(self, y):
        self._check_fitted()
        self._check_response_var()
        pass

    def _create_input_nub(self):
        # TODO
        pass

    def _create_output_nub(self):
        self._check_response_var()
        pass

    def _create_mapper(self, variable_list):
        pass

    def _check_fitted(self):
        if not self.fitted:
            raise AssertionError('Automater has not been fitted yet. Please call to Automater.fit() with appropriate '
                                 'data to fit the model. ')
        else:
            return True

    def _check_response_var(self):
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
        for outer_datatype, outer_variable_list in self.data_type_dict.items():
            for inner_datatype, inner_variable_list in self.data_type_dict.items():

                # Do not compare data types to themselves
                if inner_datatype == outer_datatype:
                    continue

                else:
                    intersection = set(outer_variable_list).intersection(set(inner_variable_list))
                    if len(intersection) > 0:
                        raise ValueError('Datatype lists {} and {} overlap, and share variables(s): {}'.
                                         format(inner_datatype, outer_datatype, intersection))

        # TODO Check that all datatype handlers are available

        if self.supervised:
            # Check that response variable is in the data_type_dict
            variable_list = reduce(lambda x, y: x + y, self.data_type_dict.values())
            if self.output_var not in variable_list:
                raise ValueError('Output variable: {} is not in variable type dict: {}. Please add output variable to '
                                 'data type dict.'.format(self.output_var, self.data_type_dict))

            # Check that respone_var 's datatype class supports output

            pass

        pass
