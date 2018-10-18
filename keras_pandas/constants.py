import logging
from collections import defaultdict

import keras
import numpy
from keras import losses
from keras.layers import Embedding, Flatten, Bidirectional, LSTM
from sklearn.preprocessing import Imputer, StandardScaler

from keras_pandas.transformations import EmbeddingVectorizer, CategoricalImputer, LabelEncoder, StringEncoder

default_sklearn_mapper_pipelines = defaultdict(lambda: list())

default_sklearn_mapper_pipelines.update({
    'numerical_vars': [Imputer(strategy='mean'), StandardScaler()],
    'categorical_vars': [StringEncoder(), CategoricalImputer(strategy='constant', fill_value='UNK', fill_unknown_labels=True),
                         LabelEncoder()],
    'text_vars': [StringEncoder(), EmbeddingVectorizer()],
    'non_transformed_vars': []
})

default_suggested_losses = {
    'numerical_vars': losses.mean_squared_error,
    'categorical_vars': losses.sparse_categorical_crossentropy,
}


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

    # TODO Convert below to numpy.max (?)
    categorical_num_levels = int(max(transformed)) + 2
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


def input_nub_text_handler(variable, input_dataframe):
    """
    Create an input nub for text data, by:

     - Finding all derived variables. With a variable `name` and sequence length of 4, there would be 4 derived
     variables, `name_0` through `name_4`
     - Creating an appropriately shaped input layer, embedding layer, and all future layers
     - Return both the input layer and last layer (both are necessary for creating a model)

    :param variable: Name of the variable
    :type variable: str
    :param input_dataframe: A dataframe, containing either the specified variable, or derived variables
    :type input_dataframe: pandas.DataFrame
    :return: A tuple containing the input layer, and the last layer of the nub
    """
    logging.info('Creating text input nub for variable: {}'.format(variable))

    # Get transformed data for shaping
    if variable in input_dataframe.columns:
        variable_list = [variable]
    else:
        variable_name_prefix = variable + '_'
        variable_list = list(filter(lambda x: x.startswith(variable_name_prefix), input_dataframe.columns))

    logging.info('Text var has variable / derived variable list: {}'.format(variable_list))
    transformed = input_dataframe[variable_list].as_matrix()

    # Set up sequence length for input_layer layer
    if len(transformed.shape) >= 2:
        input_sequence_length = int(transformed.shape[1])
    else:
        input_sequence_length = 1

    # Get the vocab size (number of rows in the embedding). The additional offsets are due to 1  for len vs indexing w/
    # 0, 1 for unknown token, and the others for something else?
    vocab_size = int(numpy.max(transformed)) + 4

    # Determine the embedding output size (number of columns in the embedding)
    # TODO There must be a better heuristic
    embedding_output_dim = 200

    logging.info('Creating embedding for text_var: {}, with input_sequence_length: {}, vocab size: {}, '
                 'and embedding_output_dim: {}'.format(variable, input_sequence_length, vocab_size,
                                                       embedding_output_dim))

    # Create & stack layers
    input_layer = keras.Input(shape=(input_sequence_length,), name='input_{}'.format(variable))

    embedding_layer = Embedding(input_dim=vocab_size,
                                output_dim=embedding_output_dim,
                                input_length=input_sequence_length, name='embedding_{}'.format(variable))

    x = embedding_layer(input_layer)
    x = Bidirectional(LSTM(128, name='lstm_{}'.format(variable)), name='bidirectiona_lstm_{}'.format(variable))(x)

    return input_layer, x


default_input_nub_type_handlers = dict()

default_input_nub_type_handlers.update({
    'numerical_vars': input_nub_numeric_handler,
    'categorical_vars': input_nub_categorical_handler,
    'text_vars': input_nub_text_handler
})
