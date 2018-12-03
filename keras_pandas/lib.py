import inspect
import logging
import os
import re
import string
import tempfile

import numpy
import pandas
import requests


def check_variable_list_are_valid(variable_type_dict):
    """
    Checks that the provided variable_type_dict is valid, by:

     - Confirming there is no overlap between all variable lists

    :param variable_type_dict: A dictionary, with keys describing variables types, and values listing particular
        variables
    :type variable_type_dict: {str:[str]}
    :return: True, if there is no overlap
    :rtype: bool
    """
    for outer_key, outer_value in variable_type_dict.items():
        for inner_key, inner_value in variable_type_dict.items():

            # Do not compare variable types to themselves
            if inner_key == outer_key:
                continue

            else:
                intersection = set(outer_value).intersection(set(inner_value))
                if len(intersection) > 0:
                    raise ValueError('Variable lists {} and {} overlap, and share key(s): {}'.
                                     format(inner_key, outer_key, intersection))

    return True


def get_variable_type(variable_name, variable_type_dict, response_var):
    # TODO This seems unnecessary. We should be able to get the variable type for any variable with this function
    if variable_name is not response_var:
        raise KeyError('Provided variable: {} not in response variable: {}'.format(variable_name,
                                                                                   response_var))

    # Filter to variable_types with the variable of interest
    variable_type_tuples = list(filter(lambda i: variable_name in i[1], variable_type_dict.items()))

    # Extract only the variable type
    variable_types = list(map(lambda i: i[0], variable_type_tuples))

    return variable_types


def download_file(url, local_file_path, filename):
    """
    Download the file at `url` in chunks, to the location at `local_file_path`

    :param url: URL to a file to be downloaded
    :type url: str
    :param local_file_path: Path to download the file to
    :type local_file_path: str
    :param filename: Filename to save the data to
    :type filename: str
    :return: The path to the file on the local machine (same as input `local_file_path`)
    :rtype: str
    """
    logging.info('Downloading file from url: {}, to path: {}'.format(url, local_file_path))
    # Reference variables
    chunk_count = 0
    local_file_path = os.path.expanduser(local_file_path)
    if not os.path.exists(local_file_path):
        os.makedirs(local_file_path)

    local_file_path = os.path.join(local_file_path, filename)

    # Open output file
    if not os.path.exists(local_file_path):
        with open(local_file_path, 'wb') as open_file:

            # Create connection to the stream
            request = requests.get(url, stream=True)

            # Iterate through chunks of file
            for chunk in request.iter_content(chunk_size=1048576):

                logging.debug('Downloading chunk: {} for file: {}'.format(chunk_count, local_file_path))

                # If there is a chunk to write to file, write it
                if chunk:
                    open_file.write(chunk)

                # Increase chunk counter
                chunk_count = chunk_count + 1

        request.close()
    return local_file_path


def get_temp_dir():
    temp_dir = tempfile.mkdtemp(prefix='python_starter')
    logging.info('Created temp_dir: {}'.format(temp_dir))
    print('Created temp_dir: {}'.format(temp_dir))
    return temp_dir


def load_titanic():
    """
    Load the titanic data set, as a pandas DataFrame

    :return: A DataFrame, containing the titanic dataset
    :rtype: pandas.DataFrame
    """
    logging.info('Loading titanic data')
    file_path = download_file('http://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv',
                              '~/.keras-pandas/example_datasets/', filename='titanic.csv')

    observations = pandas.read_csv(file_path)
    observations.columns = list(map(lambda x: x.lower().replace(' ', '_').replace('/', '_'), observations.columns))
    logging.info('Available titanic columns: {}'.format(observations.columns))
    return observations


def load_iris():
    logging.info('Loading iris data')
    file_path = download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                              '~/.keras-pandas/example_datasets/', filename='iris.csv')
    observations = pandas.read_csv(file_path, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
                                                     'class'])

    logging.info('Available iris columns: {}'.format(observations.columns))
    return observations


def load_mushroom():
    logging.info('Loading mushroom data')
    file_path = download_file(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data',
        '~/.keras-pandas/example_datasets/', filename='agaricus-lepiota.csv')
    observations = pandas.read_csv(file_path,
                                   names=['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                                          'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
                                          'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                                          'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
                                          'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'])

    logging.info('Available mushroom columns: {}'.format(observations.columns))
    return observations


def load_lending_club(test_run=True):
    logging.info('Loading lending club data')
    file_path = download_file('https://resources.lendingclub.com/LoanStats3a.csv.zip',
                              '~/.keras-pandas/example_datasets/',
                              filename='lending_club.csv.zip')
    logging.info('Reading data from filepath: {}'.format(file_path))

    observations = pandas.read_csv(file_path, compression='zip', skiprows=1, skipfooter=4, skip_blank_lines=True)

    # Coarse data transformations
    for variable in ['int_rate', 'revol_util']:
        observations[variable] = observations[variable].apply(lambda x: str(x).strip('%') if x else None)
        observations[variable] = pandas.to_numeric(observations[variable], errors='coerce')

    if test_run:
        observations = observations.sample(300)

    logging.info('Available lending club columns: {}'.format(observations.columns))
    return observations


def load_instanbul_stocks(as_ts=False):
    logging.info('Loading Instanbul data')
    file_path = download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/00247/data_akbilgic.xlsx',
                              '~/.keras-pandas/example_datasets/',
                              filename='instanbul_stocks.xlsw')
    logging.info('Reading data from filepath: {}'.format(file_path))

    observations = pandas.read_excel(file_path, header=1)
    observations.columns = list(map(lambda x: x.lower().replace(' ', '_').replace('/', '_'), observations.columns))
    logging.info('Available raw columns: {}'.format(observations.columns))

    # Coarse data transformations
    observations['date'] = pandas.to_datetime(observations['date'])

    if as_ts:
        logging.info('Reformatting data as timeseries')
        observations = observations.sort_values('date')

        lagged_vars = ['ise', 'ise.1', 'sp', 'dax']
        for lagged_var in lagged_vars:
            shifts = zip(observations[lagged_var].shift(1), observations[lagged_var].shift(2),
                         observations[lagged_var].shift(3))

            # Convert shifts to numpy object
            # shifts = map(lambda x: numpy.array(x), shifts)
            # Convert iterator to list
            shifts = list(shifts)
            observations[lagged_var + '_lagged'] = shifts

            # Convert from tuple to list
            observations[lagged_var + '_lagged'] = observations[lagged_var + '_lagged'].apply(list)

        observations = observations[3:]

    observations = observations.copy()
    logging.info('Available Istanbul columns: {}'.format(observations.columns))
    return observations


def check_valid_datatype(datatype_class):
    """
    Check whether the provided `datatype_class` meets the requirements for use as a keras-pandas datatype handler,
    using duck-typing

    :param datatype_class: A class, with the expected signature
    :return: Whether or not the datatype_class has the requirements
    :rtype: bool
    """
    datatype_attributes = inspect.getmembers(datatype_class)
    datatype_attributes = set(map(lambda x: x[0], datatype_attributes))

    logging.info('datatype: {} has attributes: {}'.format(datatype_class, datatype_attributes))

    required_input_signature = {'supports_output', 'default_transformation_pipeline', 'input_nub_generator'}
    required_output_signature = required_input_signature.union(
        {'input_nub_generator', 'output_inverse_transform', 'output_suggested_loss'})

    if hasattr(datatype_attributes, 'supports_output') and datatype_class.supports_output:
        is_valid = required_output_signature.issubset(datatype_attributes)
    else:
        is_valid = required_input_signature.issubset(datatype_attributes)

    return is_valid


def namespace_conversion(input_string):
    """
    Convert input_string to be sfve in the tensorflow namespace

    :param input_string: A string, to be converted
    :type input_string: str
    :return: Cleanly formatted version of input_string
    :rtype: str
    """
    # TODO we should check if this is in the current namespace already, and increment a number if it is
    letters = s = set(string.ascii_lowercase + string.ascii_uppercase)
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', input_string)
    if len(cleaned) <=0:
        logging.warning('Input string: {} reduced to empty variable name. Replaced w/: {}'.format(input_string, cleaned))
        cleaned = 'name_' + numpy.random.randint(0,1000)
    if cleaned[0] not in letters:
        cleaned = 'start_' + cleaned
    if cleaned[-1] not in letters:
        cleaned = cleaned + '_end'
    logging.info('input_string: {} converted to cleaned: {}'.format(input_string, cleaned))
    return cleaned
