import logging

from keras import Model
from keras.layers import Dense

from keras_pandas.Automater import Automater
from keras_pandas.lib import load_mushrooms, load_titanic


def main():
    logging.getLogger().setLevel(logging.DEBUG)

    observations = load_titanic()

    # Transform the data set, using keras_pandas
    categorical_vars = ['pclass', 'sex', 'survived']
    numerical_vars = ['age', 'siblings_spouses_aboard', 'parents_children_aboard', 'fare']
    print observations.columns
    auto = Automater(categorical_vars=categorical_vars, numerical_vars=numerical_vars, response_var='survived')
    X, y = auto.fit_transform(observations)

    # Create model
    x = auto.input_nub
    x = Dense(30)(x)
    x = auto.output_nub(x)

    model = Model(inputs=auto.input_layers, outputs=x)
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X, y, epochs=10)

    pass

if __name__ == '__main__':
    main()
