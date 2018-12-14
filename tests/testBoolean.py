from keras import Model
from sklearn_pandas import DataFrameMapper

from keras_pandas import lib
from keras_pandas.data_types.Boolean import Boolean
from tests.testbase import TestBase


class TestBoolean(TestBase):

    def test_init(self):
        # Create datatype
        datatype = Boolean()

        # Check for output support (or not)
        self.assertTrue(datatype.supports_output)

    def test_datatype_signature(self):
        # Create datatype
        datatype = Boolean()

        # Check valid datatype
        lib.check_valid_datatype(datatype)

    def test_whole(self):
        # Create datatype
        datatype = Boolean()

        # Load observations
        observations = lib.load_titanic()
        variable_name = 'survived'

        # Transform observations
        mapper = DataFrameMapper([([variable_name], datatype.default_transformation_pipeline)], df_out=True)
        transformed_df = mapper.fit_transform(observations)

        # TODO Create network
        input_layer, input_nub = datatype.input_nub_generator(variable_name, transformed_df)
        output_nub = datatype.output_nub_generator(variable_name, transformed_df)

        x = input_nub
        x = output_nub(x)

        model = Model(input_layer, x)
        model.compile(optimizer='adam', loss=datatype.output_suggested_loss())


        pass
