from keras.backend import name_scope, placeholder

from keras_pandas import lib
from keras_pandas.data_types.Abstract import AbstractDatatype
from keras_pandas.lib import check_valid_datatype
from tests.testbase import TestBase


class TestLib(TestBase):

    def test_namespace_conversion(self):
        placeholder(name=lib.namespace_conversion(' asdf @$@#$@#'))
        placeholder(name=lib.namespace_conversion('asdf @$ @#$@#'))
        placeholder(name=lib.namespace_conversion('12342342'))
        pass
