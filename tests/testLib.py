from keras.backend import placeholder

from keras_pandas import lib
from tests.testbase import TestBase


class TestLib(TestBase):

    def test_namespace_conversion(self):
        placeholder(name=lib.namespace_conversion(' asdf @$@#$@#'))
        placeholder(name=lib.namespace_conversion('asdf @$ @#$@#'))
        placeholder(name=lib.namespace_conversion('12342342'))
        iris_vars = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'class', 'Iris Setosa', 'Iris Versicolour', 'Iris Virginica']
        for var in iris_vars:
            placeholder(name=lib.namespace_conversion(var))
        pass
