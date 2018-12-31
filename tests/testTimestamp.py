from keras import Model
from keras.layers import Dense
from sklearn_pandas import DataFrameMapper

from keras_pandas import lib
from keras_pandas.data_types.Timestamp import Timestamp
from tests.testbase import TestBase


class TestTimestamp(TestBase):

    def test_init(self):
        # Create datatype
        datatype = Timestamp()

        # Check for output support (or not)
        self.assertFalse(datatype.supports_output)

    def test_datatype_signature(self):
        # Create datatype
        datatype = Timestamp()

        # Check valid datatype
        lib.check_valid_datatype(datatype)

    def test_whole(self):
        # Create datatype
        datatype = Timestamp()

        # Load observations
        observations = lib.load_lending_club()
        observations['last_pymnt_d'] = observations['last_pymnt_d'].apply(lambda x: str(x).replace('-', ' '))
        variable_name = 'last_pymnt_d'

        # Transform observations
        mapper = DataFrameMapper([([variable_name], datatype.default_transformation_pipeline)], df_out=True)
        transformed_df = mapper.fit_transform(observations)
        print(transformed_df.columns)

        # Create network
        input_layer, input_nub = datatype.input_nub_generator(variable_name, transformed_df)
        output_nub = Dense(1)

        x = input_nub
        x = output_nub(x)

        model = Model(input_layer, x)
        model.compile(optimizer='adam', loss='mse')
