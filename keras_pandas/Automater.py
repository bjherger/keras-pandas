from functools import reduce


class Automater(object):

    def __init__(self, variable_type_dict=dict(), response_var=None):

        self.variable_type_dict = variable_type_dict
        self.input_vars = reduce(lambda x, y: x+y, self.variable_type_dict.values())
        if response_var is not None:
            self.input_vars.remove(response_var)
        self.response_var = response_var
        self.supervised = self.response_var != None
        self.input_mapper = None
        self.output_mapper = None
        self.fitted = False

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
        if self.response_var is None:
            raise AssertionError('Attempting to call to function that requires a response variable. Please create a new'
                                 'automater, using the response_var parater')
        else:
            return True

    def _check_input_df(self, input_dataframe):
        # TODO Check that input_dataframe contains all variables, except for response variable
        pass

    def _valid_configurations_check(self):
        # TODO Check that each variable is assigned to only one variable type
        # TODO Check that all datatype handlers are available
        if self.supervised:
            # TODO Check that response variable is in the variable_type_dict
            # TODO Check that respone_var 's datatype class supports output
            pass

        pass
