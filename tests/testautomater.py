import copy
import logging
import unittest

import numpy
import pandas
from keras import Model
from keras.layers import Dense

from keras_pandas import lib
from keras_pandas.Automater import Automater
from tests.testbase import TestBase


class TestAutomater(TestBase):

    def test_check_variable_lists_are_valid(self):
        # Base case: No variables
        data = {
            'numerical_vars': [],
            'categorical_vars': [],
            'datetime_vars': []
        }
        self.assertEqual(True, lib.check_variable_list_are_valid(data))

        # Common use case: Variables in each
        data = {
            'numerical_vars': ['n1', 'n2', 'n3'],
            'categorical_vars': ['c1', 'c2', 'c3'],
            'datetime_vars': ['d1', 'd2']
        }
        self.assertEqual(True, lib.check_variable_list_are_valid(data))

        # Overlapping variable lists
        data = {
            'numerical_vars': ['n1', 'n2', 'n3', 'x1'],
            'categorical_vars': ['c1', 'c2', 'c3'],
            'datetime_vars': ['d1', 'd2', 'x1']
        }

        self.assertRaises(ValueError, lib.check_variable_list_are_valid, data)

        # Multiple overlapping variable lists
        data = {
            'numerical_vars': ['n1', 'n2', 'n3', 'x1'],
            'categorical_vars': ['c1', 'c2', 'c3', 'x1'],
            'datetime_vars': ['d1', 'd2', 'x1']
        }

        self.assertRaises(ValueError, lib.check_variable_list_are_valid, data)

    def test_create_sklearn_pandas_mapper_pipeline_length(self):
        # Base case: No variables
        data = {}
        input_mapper, output_mapper = Automater()._create_mappers(data)
        self.assertCountEqual(list(), input_mapper.features)
        self.assertCountEqual(list(), output_mapper.features)

        # A single numerical
        data = {'numerical_vars': ['n1']}
        input_mapper, output_mapper = Automater()._create_mappers(data)
        self.assertEqual(1, len(input_mapper.features))

        # Two numerical
        data = {'numerical_vars': ['n1', 'n2']}
        input_mapper, output_mapper = Automater()._create_mappers(data)
        self.assertEqual(2, len(input_mapper.features))

        # Two variables of different types
        data = {'numerical_vars': ['n1'],
                'categorical_vars': ['c1']}
        input_mapper, output_mapper = Automater()._create_mappers(data)
        self.assertEqual(2, len(input_mapper.features))

        # Two varibles with default pipelines
        data = {'NO_DEFAULT_ASDFSDA': ['x1', 'x2']}
        input_mapper, output_mapper = Automater()._create_mappers(data)
        self.assertEqual(2, len(input_mapper.features))

        mapper_pipelines = list(map(lambda x: list(x[1]), input_mapper.features))
        self.assertCountEqual([[], []], mapper_pipelines)

    def test_initializer(self):
        # Base case: No variables
        auto = Automater()
        self.assertEqual({'numerical_vars': list(), 'categorical_vars': list(),
                          'datetime_vars': list(), 'text_vars': list(),
                          'non_transformed_vars': list()}, auto._variable_type_dict, )
        self.assertCountEqual(list(), auto._user_provided_variables)

        # Common use case: Variables in each
        data = {
            'numerical_vars': ['n1', 'n2', 'n3'],
            'categorical_vars': ['c1', 'c2', 'c3'],
            'datetime_vars': ['d1', 'd2']
        }

        response = copy.deepcopy(data)
        response['non_transformed_vars'] = list()
        response['text_vars'] = list()

        auto = Automater(numerical_vars=data['numerical_vars'], categorical_vars=data['categorical_vars'],
                         datetime_vars=data['datetime_vars'])

        self.assertEqual(False, auto.fitted)
        self.assertEqual(response, auto._variable_type_dict)

        response_variable_list = [item for sublist in response.values() for item in sublist]
        self.assertCountEqual(response_variable_list, auto._user_provided_variables)
        # Overlapping variable lists
        data = {
            'numerical_vars': ['n1', 'n2', 'n3', 'x1'],
            'categorical_vars': ['c1', 'c2', 'c3'],
            'datetime_vars': ['d1', 'd2', 'x1']
        }

        response = copy.deepcopy(data)
        response['non_transformed_vars'] = list()

        self.assertRaises(ValueError, Automater().__init__(), numerical_vars=data['numerical_vars'],
                          categorical_vars=data['categorical_vars'],
                          datetime_vars=data['datetime_vars'])


    def test_inverse_transform_numerical_response(self):

        # :oad data
        observations = lib.load_lending_club()

        # Set to test run
        observations = observations.sample(n=100)

        # Declare variable types
        categorical_vars = ['term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status',
                            'issue_d',
                            'pymnt_plan', 'purpose', 'addr_state', 'initial_list_status', 'application_type',
                            'disbursement_method', 'loan_status']
        numerical_vars = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'annual_inc', 'installment', 'dti',
                          'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'pub_rec_bankruptcies',
                          'int_rate', 'revol_util']

        text_vars = ['desc', 'title']

        # Manual null filling
        for categorical_var in categorical_vars:
            observations[categorical_var] = observations[categorical_var].fillna('None')
            observations[categorical_var] = observations[categorical_var].apply(str)

        auto = Automater(categorical_vars=categorical_vars, numerical_vars=numerical_vars, text_vars=text_vars,
                         response_var='funded_amnt')

        X, y = auto.fit_transform(observations)

        # Start model with provided input nub
        x = auto.input_nub

        # Fill in your own hidden layers
        x = Dense(8)(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(8)(x)

        # End model with provided output nub
        x = auto.output_nub(x)

        model = Model(inputs=auto.input_layers, outputs=x)
        model.compile(optimizer='Adam', loss=auto.loss, metrics=['accuracy'])

        # Train model
        logging.warning('Settle in! This training normally takes about 5-20 minutes on CPU')
        model.fit(X, y, epochs=1, validation_split=.2)
        unscaled_preds = model.predict(X)

        logging.debug('unscaled_preds: {}'.format(list(unscaled_preds)))

        scaled_preds = auto.inverse_transform_output(unscaled_preds)

        logging.debug('scaled_preds: {}'.format(list(scaled_preds)))

        self.assertNotAlmostEquals(0, numpy.mean(scaled_preds))

        self.assertNotAlmostEquals(1, numpy.std(scaled_preds))

    def test_inverse_transform_numerical_response(self):

        # :oad data
        observations = lib.load_lending_club()

        # Set to test run
        observations = observations.sample(n=100)

        # Declare variable types
        categorical_vars = ['term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status',
                            'issue_d',
                            'pymnt_plan', 'purpose', 'addr_state', 'initial_list_status', 'application_type',
                            'disbursement_method', 'loan_status']
        numerical_vars = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'annual_inc', 'installment', 'dti',
                          'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'pub_rec_bankruptcies',
                          'int_rate', 'revol_util']

        text_vars = ['desc', 'title']

        # Manual null filling
        for categorical_var in categorical_vars:
            observations[categorical_var] = observations[categorical_var].fillna('None')
            observations[categorical_var] = observations[categorical_var].apply(str)

        auto = Automater(categorical_vars=categorical_vars, numerical_vars=numerical_vars, text_vars=text_vars,
                         response_var='funded_amnt')

        X, y = auto.fit_transform(observations)

        # Start model with provided input nub
        x = auto.input_nub

        # Fill in your own hidden layers
        x = Dense(8)(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(8)(x)

        # End model with provided output nub
        x = auto.output_nub(x)

        model = Model(inputs=auto.input_layers, outputs=x)
        model.compile(optimizer='Adam', loss=auto.loss, metrics=['accuracy'])

        # Train model
        logging.warning('Settle in! This training normally takes about 5-20 minutes on CPU')
        model.fit(X, y, epochs=1, validation_split=.2)
        unscaled_preds = model.predict(X)

        logging.debug('unscaled_preds: {}'.format(list(unscaled_preds)))

        scaled_preds = auto.inverse_transform_output(unscaled_preds)

        logging.debug('scaled_preds: {}'.format(list(scaled_preds)))

        self.assertNotAlmostEquals(0, numpy.mean(scaled_preds))

        self.assertNotAlmostEquals(1, numpy.std(scaled_preds))

