import logging

from sklearn.model_selection import train_test_split

from keras_pandas import lib
from keras_pandas.Automater import Automater


def main():
    # Load data
    observations = lib.load_lending_club()
    print('Observation columns: {}'.format(list(observations.columns)))
    print('Class balance:\n {}'.format(observations['loan_status'].value_counts()))

    # Train /test split
    train_observations, test_observations = train_test_split(observations)
    train_observations = train_observations.copy()
    test_observations = test_observations.copy()

    # List out variable types

    numerical_vars = ['loan_amnt', 'annual_inc', 'open_acc', 'dti', 'delinq_2yrs',
                      'inq_last_6mths', 'mths_since_last_delinq', 'pub_rec', 'revol_bal', 'revol_util',
                      'total_acc', 'pub_rec_bankruptcies']
    categorical_vars = ['term', 'grade', 'emp_length', 'home_ownership', 'loan_status', 'addr_state',
                        'application_type', 'disbursement_method']
    text_vars = ['desc', 'purpose', 'title']
    response_var = 'loan_status'

    # Create and fit Automater
    auto = Automater(data_type_dict=data_type_dict, output_var=None)
    auto.fit(train_observations)

    # TODO Create and fit keras (deep learning) model.
    # TODO List out which components are supplied by Automater
    # TODO Inverse transform model output, to get usable results
    # TODO Save all results

    pass

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()