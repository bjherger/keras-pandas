import logging

from keras import Model
from keras.layers import Dense

from keras_pandas.Automater import Automater
from keras_pandas.lib import load_mushrooms


def main():
    logging.getLogger().setLevel(logging.DEBUG)

    observations = load_mushrooms()

    # Transform the data set, using keras_pandas
    auto = Automater(categorical_vars=observations.columns, response_var='class')
    X, y = auto.fit_transform(observations)

    # Create model
    x = auto.input_nub
    x = Dense(30)(x)
    x = auto.output_nub(x)

    model = Model(inputs=auto.input_layers, outputs=x)
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy')

    # Train model
    model.fit(X, y)

    pass

if __name__ == '__main__':
    main()
