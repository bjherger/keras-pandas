# keras-pandas

**tl;dr:** keras-pandas allows users to rapidly build and iterate on deep learning models. 

Getting data formatted and into keras can be tedious, time consuming, and difficult, whether your a veteran or new to 
Keras. `keras-pandas` overcomes these issues by (automatically) providing:

 - A cleaned, transformed and correctly formatted `X` and `y` (good for keras, sklearn or any other ML platform)
 - An 'input nub', without the hassle of worrying about input shapes or data types
 - An output layer, correctly formatted for the kind of response variable provided
 
With these resources, it's possible to rapidly build and iterate on deep learning models, and focus on the parts of 
modeling that you enjoy!

## Quick Start

Let's build a model with the [titanic data set](https://www.kaggle.com/c/titanic/data. This data set is particularly 
fun because this data set contains a mix of categorical and numerical data types, and features a lot of null values. 

We'll `keras-pandas`

```bash
pip install -U keras-pandas
```

And then run the following snippet to create and train a model:

```python
from keras import Model
from keras.layers import Dense

from keras_pandas.Automater import Automater
from keras_pandas.lib import load_titanic

# Load the titanic data set, as a pandas dataframe
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
response variable (optional, for supervised problems). Together, all of these variables are the `user_input_variables`, 
which may be different than the variables fed into Keras. 

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
omit the response variable.

```python
categorical_vars = ['pclass', 'sex', 'survived']
numerical_vars = ['age', 'siblings_spouses_aboard', 'parents_children_aboard', 'fare']

auto = Automater(categorical_vars=categorical_vars, numerical_vars=numerical_vars)
```

In this case, an output nub will not be auto-generated

### Fitting the Automater

Before use, the `Automator` must be fit. The `fit()` method accepts a pandas DataFrame, which must contain all of the 
columns listed during initialization.

```python
auto.fit(observations)
```

### Transforming data

Now, we can use our `Automater` to transform the dataset, from a pandas DataFrame to numpy objects properly formatted
for Keras's input and output layers. 

```python
X, y = auto.transform(observations, df_out=False)
```

This will return two objects:

  - `X`: An array, containing numpy object for each Keras input. This is generally one Keras input for each user 
  input variable. 
  - `y`: A numpy object, containing the response variable (if one was provided) 

### Using input / output nubs

Setting up correctly formatted, heuristically 'good' input and output layers is often

 - Tedious
 - Time consuming
 - Difficult for those new to Keras
 
With this in mind, `keras-pandas` provides correctly formatted input and output 'nubs'. 

The input nub is correctly formatted to accept the output from `auto.transform()`. It contains one Keras Input layer 
for each generated input, may contain addition layers, and has all input piplines joined with a `Concatenate` layer. 

The output layer is correctly formatted to accept the response variable numpy object.  

## Contributing

If you're interested in helping out, all open tasks are listed the GitHub Issues tab. The issues tagged with 
`first issue` are a good place to start if your new to the project or new to open source projects. 

If you're interested in a new major feature, please feel free to reach out to me

### Bug reports

The best bug reports are Pull Requests. The second best bug reports are new issues on this repo.

### Test

This framework uses `unittest` for unit testing. Tests can be run by calling:

```bash
cd tests/

python -m unittest discover -s . -t .
```
### Style guide

This codebase should follow [Google's Python Style Guide](https://google.github.io/styleguide/pyguide.html). 

### Generating documentation

This codebase uses [sphinx](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)'s 
[autodoc](http://www.sphinx-doc.org/en/master/ext/autodoc.html) feature. To generate new documentation, to reflect 
updated documentation, run:

```bash
cd docs

make html

```  

## Contact

Hey, I'm Brendan Herger, avaiable at [https://www.hergertarian.com/](https://www.hergertarian.com/). Please feel free 
to reach out to me at `13herger <at> gmail <dot> com`
