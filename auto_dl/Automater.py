class Automater(object):
    def __init__(self, numerical_vars=list(), categorical_vars=list(), boolean_vars=list(),
                 non_transformed_vars=list()):

        # TODO
        self._sklearn_pandas_object = None

        # TODO
        self._datetime_expansion_method_dict = None

        # TODO
        self._embedding_size_function = None

        # TODO
        self._variable_transformer_dict = None

        # TODO
        self._input_variables = None

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
