from collections import defaultdict

import keras
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder

default_sklearn_mapper_pipelines = defaultdict(lambda: None)

default_sklearn_mapper_pipelines.update({
    'numerical_vars': [Imputer(strategy='mean'), StandardScaler()],
    'categorical_vars': [LabelEncoder()],
    'boolean_vars': [LabelEncoder()],
    'non_transformed_vars': None
})


def input_nub_numeric_handler(variable, input_dataframe):
    # Get transformed data for shaping
    transformed = input_dataframe[variable].as_matrix()

    # Set up dimensions for input_layer layer
    if len(transformed.shape) >= 2:
        input_length = int(transformed.shape[1])
    else:
        input_length = 1

    # Create input_layer layer
    input_layer = keras.Input(shape=(input_length,), dtype='float32', name='input_{}'.format(variable))

    # Return, in format of input_layer, last variable-specific layer
    return input_layer, input_layer


default_input_nub_type_handlers = dict()

default_input_nub_type_handlers.update({
    'numerical_vars': input_nub_numeric_handler
})
