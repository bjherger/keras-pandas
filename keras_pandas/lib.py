import logging
import os
import pandas

import requests


def check_variable_list_are_valid(variable_type_dict):
    """

    :param variable_type_dict:
    :return:
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

def download_file(url, local_file_path, filename):
    """
    Download the file at `url` in chunks, to the location at `local_file_path`
    :param url: URL to a file to be downloaded
    :type url: str
    :param local_file_path: Path to download the file to
    :type local_file_path: str
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

    # Create connection to the stream
    r = requests.get(url, stream=True)

    # Open output file
    if not os.path.exists(local_file_path):
        with open(local_file_path, 'wb') as f:

            # Iterate through chunks of file
            for chunk in r.iter_content(chunk_size=1048576):

                logging.debug('Downloading chunk: {} for file: {}'.format(chunk_count, local_file_path))

                # If there is a chunk to write to file, write it
                if chunk:
                    f.write(chunk)

                # Increase chunk counter
                chunk_count = chunk_count + 1

        r.close()
    return local_file_path

def load_mushrooms():
    # Extract the data
    file_path = download_file(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data',
        '~/.keras-pandas/example_datasets/', filename='agaricus-lepiota.data')

    observations = pandas.read_csv(filepath_or_buffer=file_path,
                                   names=['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                                          'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                                          'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                                          'stalk-surface-below-ring', 'stalk-color-above-ring',
                                          'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                                          'ring-type', 'spore-print-color', 'population', 'habitat'])
    return observations

def load_titanic():
    file_path = download_file('http://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv',
                              '~/.keras-pandas/example_datasets/', filename='titanic.csv')

    observations = pandas.read_csv(file_path)
    observations.columns = map(lambda x: x.lower().replace(' ', '_').replace('/', '_'), observations.columns)

    return observations