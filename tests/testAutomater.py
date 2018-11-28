import pandas
from functools import reduce

import numpy
from keras import Model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper

from keras_pandas import lib
from keras_pandas.Automater import Automater
from tests.testbase import TestBase


class TestAutomater(TestBase):

    def test_bad_init_config(self):
        # Bad configuration: Supervised
        data_type_dict = {'numerical': ['n1', 'n2'],
                          'categorical': ['b1', 'c1'],
                          'boolean': ['b1']}

        self.assertRaises(ValueError, Automater, data_type_dict=data_type_dict,
                          output_var='c1')

        # Bad configuration: Unsupervised
        self.assertRaises(ValueError, Automater, data_type_dict=data_type_dict)

        pass

    def test_supervised(self):
        observations = lib.load_lending_club()

        # Train /test split
        train_observations, test_observations = train_test_split(observations)
        train_observations = train_observations.copy()
        test_observations = test_observations.copy()

        # Supervised
        data_type_dict = {'numerical': ['loan_amnt', 'annual_inc', 'open_acc', 'dti', 'delinq_2yrs',
                                        'inq_last_6mths', 'mths_since_last_delinq', 'pub_rec', 'revol_bal',
                                        'revol_util',
                                        'total_acc', 'pub_rec_bankruptcies'],
                          'categorical': ['term', 'grade', 'emp_length', 'home_ownership', 'loan_status', 'addr_state',
                                          'application_type', 'disbursement_method'],
                          'text': ['desc', 'purpose', 'title']}
        output_var = 'loan_status'

        auto = Automater(data_type_dict=data_type_dict,
                         output_var=output_var)

        self.assertTrue(auto.supervised)
        expected_input_vars = reduce(lambda x, y: x + y, data_type_dict.values())
        expected_input_vars.remove(output_var)
        self.assertCountEqual(expected_input_vars, auto.input_vars)
        self.assertEqual(output_var, auto.output_var)
        self.assertTrue(isinstance(auto.input_mapper, DataFrameMapper))
        self.assertTrue(isinstance(auto.output_mapper, DataFrameMapper))
        self.assertFalse(auto.fitted)
        self.assertRaises(AssertionError, auto._check_fitted)

        # Test fit
        auto.fit(train_observations)
        self.assertTrue(auto.fitted)

        self.assertIsNotNone(auto.input_mapper.built_features)
        self.assertTrue(isinstance(auto.input_layers, list))
        self.assertEqual(len(expected_input_vars), len(auto.input_layers))
        self.assertIsNotNone(auto.input_nub)

        self.assertIsNotNone(auto.output_nub)
        self.assertIsNotNone(auto.output_mapper.built_features)

        # Test transform, df_out=False
        train_X, train_y = auto.transform(train_observations)
        test_X, test_y = auto.transform(test_observations)
        self.assertTrue(isinstance(test_X, list))
        self.assertTrue(isinstance(test_y, numpy.ndarray))
        self.assertEqual(test_observations.shape[0], test_X[0].shape[0])  # Correct number of rows back
        self.assertEqual(test_observations.shape[0], test_y.shape[0])  # Correct number of rows back

        # Test transform, df_out=True
        transformed_observations = auto.transform(test_observations, df_out=True)
        self.assertTrue(isinstance(transformed_observations, pandas.DataFrame))
        self.assertEqual(test_observations.shape[0], transformed_observations.shape[0])  # Correct number of rows back

        # Test suggest_loss
        suggested_loss = auto.suggest_loss()
        self.assertTrue(callable(suggested_loss))

        # Test model building

        x = auto.input_nub
        x = Dense(32)(x)
        x = auto.output_nub(x)

        model = Model(inputs=auto.input_layers, outputs=x)
        model.compile(optimizer='Adam', loss=auto.suggest_loss())
        model.fit(train_X, train_y)

        pred_y = model.predict(test_X)

        # Test inverse_transform_output
        inv_transformed_pred_y = auto.inverse_transform_output(pred_y)
        self.assertEqual(test_observations.shape[0], inv_transformed_pred_y.shape[0])

    def test_unsupervised(self):
        observations = lib.load_lending_club()

        # Train /test split
        train_observations, test_observations = train_test_split(observations)
        train_observations = train_observations.copy()
        test_observations = test_observations.copy()

        # Unsupervised
        data_type_dict = {'numerical': ['loan_amnt', 'annual_inc', 'open_acc', 'dti', 'delinq_2yrs',
                                        'inq_last_6mths', 'mths_since_last_delinq', 'pub_rec', 'revol_bal',
                                        'revol_util',
                                        'total_acc', 'pub_rec_bankruptcies'],
                          'categorical': ['term', 'grade', 'emp_length', 'home_ownership', 'loan_status', 'addr_state',
                                          'application_type', 'disbursement_method'],
                          'text': ['desc', 'purpose', 'title']}
        auto = Automater(data_type_dict=data_type_dict)
        self.assertFalse(auto.supervised)

        expected_input_vars = reduce(lambda x, y: x + y, data_type_dict.values())
        self.assertCountEqual(expected_input_vars, auto.input_vars)
        self.assertEqual(None, auto.output_var)
        self.assertTrue(isinstance(auto.input_mapper, DataFrameMapper))
        self.assertIsNone(auto.output_mapper)
        self.assertFalse(auto.fitted)

        self.assertRaises(AssertionError, auto._check_has_response_var)

        # Test fit
        auto.fit(train_observations)
        self.assertTrue(auto.fitted)

        self.assertIsNotNone(auto.input_mapper.built_features)
        self.assertTrue(isinstance(auto.input_layers, list))
        self.assertEqual(len(expected_input_vars), len(auto.input_layers))
        self.assertIsNotNone(auto.input_nub)

        self.assertIsNone(auto.output_nub)
        self.assertIsNone(auto.output_mapper)

        # Test transform, df_out=False
        X, y = auto.transform(test_observations)
        self.assertTrue(isinstance(X, list))
        self.assertIsNone(y)
        self.assertEqual(test_observations.shape[0], X[0].shape[0])  # Correct number of rows back

        # Test transform, df_out=True
        transformed_observations = auto.transform(test_observations, df_out=True)
        self.assertTrue(isinstance(transformed_observations, pandas.DataFrame))
        self.assertEqual(test_observations.shape[0], transformed_observations.shape[0])  # Correct number of rows back
