import logging

import numpy
import pandas
from keras import Model
from keras.layers import Dense
from sklearn.model_selection import train_test_split

from keras_pandas import lib
from keras_pandas.Automater import Automater


def main():

    # Load data
    observations = lib.load_lending_club()
    # observations = lib.load_lending_club(test_run=False)
    print('Observation columns: {}'.format(list(observations.columns)))
    print('Class balance:\n {}'.format(observations['loan_status'].value_counts()))


    # Heuristic data transformations
    for var in ['int_rate', 'revol_util']:

        # Strip out percent signs
        observations[var] = observations[var].apply(lambda x: str(x).replace('%', ''))
        observations[var] = pandas.to_numeric(observations[var], errors='coerce')
    for var in ['mths_since_last_delinq', 'annual_inc_joint']:

        # Heuristic null filling for some variables
        observations[var] = observations[var].fillna(0)

    # List out variable types
    numerical_vars = ['loan_amnt', 'annual_inc', 'open_acc', 'dti', 'delinq_2yrs',
                      'inq_last_6mths', 'mths_since_last_delinq', 'pub_rec', 'revol_bal', 'revol_util',
                      'total_acc', 'pub_rec_bankruptcies']
    categorical_vars = ['term', 'grade', 'emp_length', 'home_ownership', 'loan_status', 'addr_state',
                        'application_type', 'disbursement_method']
    text_vars = ['desc', 'purpose', 'title']

    train_observations, test_observations = train_test_split(observations)
    train_observations = train_observations.copy()
    test_observations = test_observations.copy()

    # Create and fit Automater
    auto = Automater(numerical_vars=numerical_vars, categorical_vars=categorical_vars, text_vars=text_vars,
                     response_var='loan_status')
    auto.fit(train_observations)

    # Create and fit keras (deep learning) model
    # The auto.transform, auto.input_nub, auto.input_layers, and auto.loss are provided by keras-pandas, and
    # everything else is core Keras
    train_X, train_y = auto.transform(train_observations)
    test_X, test_y = auto.transform(test_observations)

    x = auto.input_nub
    x = Dense(32)(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32)(x)
    x = auto.output_nub(x)

    model = Model(inputs=auto.input_layers, outputs=x)
    model.compile(optimizer='Adam', loss=auto.loss, metrics=['accuracy'])

    model.fit(train_X, train_y)

    test_y_pred = model.predict(test_X)

    # Inverse transform model output, to get usable results and save all results
    test_observations[auto.response_var + '_pred'] = auto.inverse_transform_output(test_y_pred)
    print('Predictions: {}'.format(test_observations[auto.response_var + '_pred']))

    pass


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()