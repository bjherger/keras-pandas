# keras-pandas

keras-pandas allows us to rapidly get a baseline model an iterate, by (automatically) providing:

 - A cleaned, transformed and correctly formatted `X` and `y`
 - A smart baseline 'input nub', without the hassle of worrying about input shapes or data types
 - A smart baseline output layer

## Quick Start

Let's say we're looking at the [titanic data set](https://www.kaggle.com/c/titanic/data), and wanted to train a model. 
This data set is particularly fun because this data set contains a mix of variables types, and features a lot of null 
values. 

We could install `keras-pandas`

```bash
pip install -U keras-pandas
```

And then run the following snippet to create and train a model:

```python
# Import a few things
from keras import Model
from keras.layers import Dense

from keras_pandas.Automater import Automater
from keras_pandas.lib import load_titanic

# Load the data set
observations = load_titanic()

# Transform the data set, using keras_pandas
categorical_vars = ['pclass', 'sex', 'survived']
numerical_vars = ['age', 'siblings_spouses_aboard', 'parents_children_aboard', 'fare']

auto = Automater(categorical_vars=categorical_vars, numerical_vars=numerical_vars, response_var='survived')
X, y = auto.fit_transform(observations)

# Create model, using the auto-generated input and output layers
x = auto.input_nub
x = Dense(30)(x)
x = auto.output_nub(x)

model = Model(inputs=auto.input_layers, outputs=x)
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=10, validation_split=.5)
``` 



## Usage

### Installation

You can install `keras-pandas` with `pip`:

```bash
pip install -U keras-pandas
```

### Creating an Automater

The core feature of `keras-pandas` is the Automater, which accepts lists of variable types (all optional), and a 
response variable (optional, for supervised problems).

As a side note, the response variable must be in one of the variable type lists (e.g. `survived` is in `categorical_vars`)

#### One variable type

If you only have one variable type, only use that variable type!
```python
categorical_vars = ['pclass', 'sex', 'survived']
auto = Automater(categorical_vars=categorical_vars, response_var='survived')
```

#### Multiple variable types
If you have multiple variable types, throw them all in!

```python
categorical_vars = ['pclass', 'sex', 'survived']
numerical_vars = ['age', 'siblings_spouses_aboard', 'parents_children_aboard', 'fare']

auto = Automater(categorical_vars=categorical_vars, numerical_vars=numerical_vars, response_var='survived')
```

#### No `response_var`

If all variables are always available, and / or your problems space doesn't have a single response variable, you can 
omit the response variable

```python
categorical_vars = ['pclass', 'sex', 'survived']
numerical_vars = ['age', 'siblings_spouses_aboard', 'parents_children_aboard', 'fare']

auto = Automater(categorical_vars=categorical_vars, numerical_vars=numerical_vars)
```

In this case, an output nub will not be auto-generated

### Fitting the Automater

Before use, the `Automator` must be fit. The `fit()` method accepts a pandas DataFrame, which must contain all of the 
columns listed upon initialization.

```python
auto.fit(observations)
```

### Transforming data

TODO

### Using input / output nubs

TODO

## Contributing

### Test

TODO

### Style guide

TODO


