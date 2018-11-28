from keras import Model
from keras.layers import Dense
from sklearn_pandas import DataFrameMapper

from keras_pandas import lib
from keras_pandas.data_types.Text import Text
from tests.testbase import TestBase


class TestNumerical(TestBase):

    def test_init(self):
        # Create datatype
        datatype = Text()

        # Check for output support (or not)
        self.assertFalse(datatype.supports_output)

    def test_datatype_signature(self):
        # Create datatype
        datatype = Text()

        # Check valid datatype
        lib.check_valid_datatype(datatype)

    def test_whole(self):
        datatype = Text()

        # Load observations
        observations = lib.load_titanic()

        # Transform observations
        mapper = DataFrameMapper([(['name'], datatype.default_transformation_pipeline),
                                  (['fare'], None)], df_out=True)
        transformed_df = mapper.fit_transform(observations)

        # Create network
        input_layer, input_nub = datatype.input_nub_generator('name', transformed_df)
        output_nub = Dense(1)

        x = input_nub
        x = output_nub(x)

        model = Model(input_layer, x)
        model.compile(optimizer='adam', loss='mse')
        pass
