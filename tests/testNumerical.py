from keras import Model
from sklearn_pandas import DataFrameMapper

from keras_pandas import lib
from keras_pandas.data_types.Numerical import Numerical
from tests.testbase import TestBase


class TestNumerical(TestBase):

    def test_init(self):
        datatype = Numerical()
        self.assertTrue(datatype.supports_output)

    def test_datatype_signature(self):
        datatype = Numerical()
        lib.check_valid_datatype(datatype)
        self.assertTrue(datatype.supports_output)

    def test_whole(self):
        # Create datatype
        datatype = Numerical()

        # Load observations
        observations = lib.load_titanic()

        # Transform observations
        mapper = DataFrameMapper([(['fare'], datatype.default_transformation_pipeline)], df_out=True)
        transformed_df = mapper.fit_transform(observations)

        # Create network
        input_layer, input_nub = datatype.input_nub_generator('fare', transformed_df)
        output_nub = datatype.output_nub_generator('fare', transformed_df)

        x = input_nub
        x = output_nub(x)

        model = Model(input_layer, x)
        model.compile(optimizer='adam', loss=datatype.output_suggested_loss())
