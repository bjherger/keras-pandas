from collections import defaultdict

from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder

mapper_default_pipelines = defaultdict(lambda: None)

mapper_default_pipelines.update({
    'numerical_vars': [Imputer(strategy='mean'), StandardScaler()],
    'categorical_vars': [LabelEncoder()],
    'boolean_vars': [LabelEncoder()],
    'non_transformed_vars': None
})

