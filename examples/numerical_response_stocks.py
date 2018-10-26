import logging

from keras import Model
from keras.layers import Dense
from sklearn.model_selection import train_test_split

from keras_pandas import lib
from keras_pandas.Automater import Automater


def main():
    # Load data
    observations = lib.load_instanbul_stocks(as_ts=True)
    print('Observation columns: {}'.format(list(observations.columns)))

    # Heuristic data transformations

    # Train /test split
    train_observations, test_observations = train_test_split(observations)
    train_observations = train_observations.copy()
    test_observations = test_observations.copy()

    # List out variable types
    timeseries_vars = ['ise_lagged', 'ise.1_lagged', 'sp_lagged', 'dax_lagged']
    numerical_vars = ['ise']

    # Create and fit Automater
    auto = Automater(numerical_vars=numerical_vars, timeseries_vars=timeseries_vars,
                     response_var='ise')
    auto.fit(train_observations)

    # Create and fit keras (deep learning) model.
    # The auto.transform, auto.input_nub, auto.input_layers, auto.output_nub, and auto.loss are provided by
    # keras-pandas, and everything else is core Keras

    x = auto.input_nub
    x = Dense(16)(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(16)(x)
    x = auto.output_nub(x)

    model = Model(inputs=auto.input_layers, outputs=x)
    model.compile(optimizer='adam', loss=auto.loss)

    train_X, train_y = auto.transform(train_observations)
    model.fit(train_X, train_y)

    # Inverse transform model output, to get usable results
    test_X, test_y = auto.transform(test_observations)
    test_y_pred = model.predict(test_X)
    test_observations[auto.response_var + '_pred'] = auto.inverse_transform_output(test_y_pred)
    print('Predictions: {}'.format(test_observations[auto.response_var + '_pred']))

    # TODO Save all results

    pass


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
