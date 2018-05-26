import logging
from collections import defaultdict

import keras
from keras.layers import Embedding, Flatten
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder


default_sklearn_mapper_pipelines = defaultdict(lambda: list())

default_sklearn_mapper_pipelines.update({
    'numerical_vars': [Imputer(strategy='mean'), StandardScaler()],
    'categorical_vars': [LabelEncoder()],
    'boolean_vars': [LabelEncoder()],
    'non_transformed_vars': []
})


def input_nub_numeric_handler(variable, input_dataframe):
    # Get transformed data for shaping
    transformed = input_dataframe[variable].as_matrix()

    # Set up dimensions for input_layer layer
    if len(transformed.shape) >= 2:
        input_sequence_length = int(transformed.shape[1])
    else:
        input_sequence_length = 1



    # Create input_layer layer
    input_layer = keras.Input(shape=(input_sequence_length,), dtype='float32', name='input_{}'.format(variable))

    # Return, in format of input_layer, last variable-specific layer
    return input_layer, input_layer

def input_nub_categorical_handler(variable, input_dataframe):
    # Get transformed data for shaping
    transformed = input_dataframe[variable].as_matrix()

    # Set up dimensions for input_layer layer
    if len(transformed.shape) >= 2:
        input_sequence_length = int(transformed.shape[1])
    else:
        input_sequence_length = 1

    categorical_num_levels = int(max(transformed)) + 1
    embedding_output_dim = int(min((categorical_num_levels + 1) / 2, 50))

    logging.info('Creating embedding for cat_var: {}, with input_sequence_length: {}, categorical_num_levels: {}, '
                 'and embedding_output_dim: {}'.format(variable, input_sequence_length, categorical_num_levels,
                                                       embedding_output_dim))

    input_layer = keras.Input(shape=(input_sequence_length,), name='input_{}'.format(variable))

    embedding_layer = Embedding(input_dim=categorical_num_levels,
                                output_dim=embedding_output_dim,
                                input_length=input_sequence_length, name='embedding_{}'.format(variable))
    embedded_sequences = embedding_layer(input_layer)
    embedding_flattened = Flatten(name='flatten_embedding_{}'.format(variable))(embedded_sequences)

    return input_layer, embedding_flattened


default_input_nub_type_handlers = dict()

default_input_nub_type_handlers.update({
    'numerical_vars': input_nub_numeric_handler,
    'categorical_vars': input_nub_categorical_handler
})
