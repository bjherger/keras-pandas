from collections import defaultdict

from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder

default_sklearn_mapper_pipelines = defaultdict(lambda: None)

default_sklearn_mapper_pipelines.update({
    'numerical_vars': [Imputer(strategy='mean'), StandardScaler()],
    'categorical_vars': [LabelEncoder()],
    'boolean_vars': [LabelEncoder()],
    'non_transformed_vars': None
})


def input_nub_numeric_handler(variable_name):
    pass

default_input_nub_type_handlers = dict()

default_input_nub_type_handlers.update({
   'numerical_vars': input_nub_numeric_handler
})


