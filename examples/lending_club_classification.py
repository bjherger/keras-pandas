import logging

from keras import Model
from keras.layers import Dense

from keras_pandas.Automater import Automater
from keras_pandas.lib import load_lending_club


def main():
    logging.getLogger().setLevel(logging.INFO)

    # Reference variables
    test_run = True

    observations = load_lending_club()

    if test_run:
        observations = observations.sample(n=100)

    # Transform the data set, using keras_pandas
    categorical_vars = ['term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 'issue_d',
                        'pymnt_plan', 'purpose', 'addr_state', 'initial_list_status', 'application_type',
                        'disbursement_method', 'loan_status']
    numerical_vars = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'annual_inc', 'installment', 'dti',
                      'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'pub_rec_bankruptcies',
                      'int_rate', 'revol_util']
    text_vars = ['desc', 'title']

    for categorical_var in categorical_vars:
        observations[categorical_var] = observations[categorical_var].fillna('None')
        observations[categorical_var] = observations[categorical_var].apply(str)

    auto = Automater(categorical_vars=categorical_vars, numerical_vars=numerical_vars, text_vars=text_vars,
                     response_var='loan_status')

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

    pass


if __name__ == '__main__':
    main()
