from keras import Model

from keras_pandas import lib
from keras_pandas.data_types.Categorical import Categorical
from keras_pandas.data_types.Numerical import Numerical
from tests.testbase import TestBase


class TestCategorical(TestBase):

    def test_init(self):
        datatype = Numerical()
        self.assertTrue(datatype.supports_output)

    def test_datatype_signature(self):
        datatype = Categorical()
        lib.check_valid_datatype(datatype)

    def test_whole(self):
        datatype = Numerical()
        observations = lib.load_titanic()

        self.assertTrue(datatype.supports_output)
        self.assertIn('default_transformation_pipeline', datatype.__dict__.keys())

        input_layer, input_nub = datatype.input_nub_generator('fare', observations)

        output_nub = datatype.output_nub_generator('fare', observations)

        x = input_nub
        x = output_nub(x)

        model = Model(input_nub, x)
        model.compile(optimizer='adam', loss='mse')
