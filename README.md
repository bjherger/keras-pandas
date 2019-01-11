# keras-pandas

[![CircleCI](https://circleci.com/gh/bjherger/keras-pandas.svg?style=svg)](https://circleci.com/gh/bjherger/keras-pandas)
[![Documentation Status](https://readthedocs.org/projects/keras-pandas/badge/?version=latest)](https://keras-pandas.readthedocs.io/en/latest/?badge=latest)

**tl;dr:** keras-pandas allows users to rapidly build and iterate on deep learning models. 

Getting data formatted and into keras can be tedious, time consuming, and require domain expertise, whether your a 
veteran or new to Deep Learning. `keras-pandas` overcomes these issues by (automatically) providing:

 - **Data transformations**: A cleaned, transformed and correctly formatted `X` and `y` (good for keras, sklearn or any 
 other ML 
 platform)
 - **Data piping**: Correctly formatted keras input, hidden and output layers to quickly start iterating on  

These approaches are build on best in world approaches from practitioners, kaggle grand masters, papers, blog posts, 
and coffee chats, to simple entry point into the world of deep learning, and a strong foundation for deep learning 
experts.  

For more info, check out the:

 - [Code](https://github.com/bjherger/keras-pandas)
 - [Documentation](http://keras-pandas.readthedocs.io/en/latest/intro.html)
 - [PyPi](https://pypi.org/project/keras-pandas/)
 - [Issue tracker](https://github.com/bjherger/keras-pandas/issues)
 - [CI/CD](https://circleci.com/gh/bjherger/keras-pandas/tree/master)
 - [Author's website](https://www.hergertarian.com/)

## Quick Start

Let's build a model with the [lending club data set](https://www.lendingclub.com/info/download-data.action). This data set is 
particularly fun because this data set contains a mix of text, categorical and numerical data types, and features a 
lot of null values. 

```bash
pip install --upgrade keras-pandas
```

```python
from keras import Model
from keras_pandas import lib
from keras_pandas.Automater import Automater
from sklearn.model_selection import train_test_split

# Load data
observations = lib.load_lending_club()

# Train /test split
train_observations, test_observations = train_test_split(observations)
train_observations = train_observations.copy()
test_observations = test_observations.copy()

# List out variable types

data_type_dict = {'numerical': ['loan_amnt', 'annual_inc', 'open_acc', 'dti', 'delinq_2yrs',
                                'inq_last_6mths', 'mths_since_last_delinq', 'pub_rec', 'revol_bal',
                                'revol_util',
                                'total_acc', 'pub_rec_bankruptcies'],
                  'categorical': ['term', 'grade', 'emp_length', 'home_ownership', 'loan_status', 'addr_state',
                                  'application_type', 'disbursement_method'],
                  'text': ['desc', 'purpose', 'title']}
output_var = 'loan_status'

# Create and fit Automater
auto = Automater(data_type_dict=data_type_dict, output_var=output_var)
auto.fit(train_observations)

# Transform data
train_X, train_y = auto.fit_transform(train_observations)
test_X, test_y = auto.transform(test_observations)

# Create and fit keras (deep learning) model.
x = auto.input_nub
x = auto.output_nub(x)

model = Model(inputs=auto.input_layers, outputs=x)
model.compile(optimizer='adam', loss=auto.suggest_loss())
```

And that's it! In a couple of lines, we've created a model that accepts a few dozen variables, and can create a world
 class deep learning model

## Usage

### Installation

You can install `keras-pandas` with `pip`:

```bash
pip install -U keras-pandas
```

### Creating an Automater

The `Automater` object is the central object in `keras-pandas`. It accepts a dictionary of the format `{'datatype': 
['var1', var2']}`

For example we could create an automater using the built in `numerical`, `categorical`, and `text` datatypes, by 
calling: 

```python
# List out variable types
data_type_dict = {'numerical': ['loan_amnt', 'annual_inc', 'open_acc', 'dti', 'delinq_2yrs',
                                'inq_last_6mths', 'mths_since_last_delinq', 'pub_rec', 'revol_bal',
                                'revol_util',
                                'total_acc', 'pub_rec_bankruptcies'],
                  'categorical': ['term', 'grade', 'emp_length', 'home_ownership', 'loan_status', 'addr_state',
                                  'application_type', 'disbursement_method'],
                  'text': ['desc', 'purpose', 'title']}
output_var = 'loan_status'

# Create and fit Automater
auto = Automater(data_type_dict=data_type_dict, output_var=output_var)
```

As a side note, the response variable must be in one of the variable type lists (e.g. `loan_status` is in `categorical_vars`)

#### One variable type

If you only have one variable type, only use one variable type!

```python
# List out variable types
data_type_dict = {'categorical': ['term', 'grade', 'emp_length', 'home_ownership', 'loan_status', 'addr_state',
                                  'application_type', 'disbursement_method']}
output_var = 'loan_status'

# Create and fit Automater
auto = Automater(data_type_dict=data_type_dict, output_var=output_var)
```

#### Multiple variable types

If you have multiple variable types, feel free to use all of them! Built in datatypes are listed in `Automater.datatype_handlers`

```python
# List out variable types
data_type_dict = {'numerical': ['loan_amnt', 'annual_inc', 'open_acc', 'dti', 'delinq_2yrs',
                                'inq_last_6mths', 'mths_since_last_delinq', 'pub_rec', 'revol_bal',
                                'revol_util',
                                'total_acc', 'pub_rec_bankruptcies'],
                  'categorical': ['term', 'grade', 'emp_length', 'home_ownership', 'loan_status', 'addr_state',
                                  'application_type', 'disbursement_method'],
                  'text': ['desc', 'purpose', 'title']}
output_var = 'loan_status'

# Create and fit Automater
auto = Automater(data_type_dict=data_type_dict, output_var=output_var)
```

#### Custom datatypes

If there's a specific datatype you'd like to use that's not built in (such as images, videos, or geospatial), you can 
include it by using `Automater`'s `datatype_handlers` parameter. 

A template datatype can be found in `keras_pandas/data_types/Abstract.py`. Filling out this template will yield a new
 datatype handler. If you're happy with your work and want to share your new datatype handler, create a PR (and check
  out `contributing.md`)
   
#### No `output_var`

If your model doesn't need a response var, or your use case doesn't use `keras-pandas`'s output functionality, you 
can skip the `output_var` by setting it to None

```python
# List out variable types
data_type_dict = {'categorical': ['term', 'grade', 'emp_length', 'home_ownership', 'loan_status', 'addr_state',
                                  'application_type', 'disbursement_method']}
output_var = None

# Create and fit Automater
auto = Automater(data_type_dict=data_type_dict, output_var=output_var)
```

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


## Contact

Hey, I'm Brendan Herger, avaiable at [https://www.hergertarian.com/](https://www.hergertarian.com/). Please feel free 
to reach out to me at `13herger <at> gmail <dot> com`

I enjoy bridging the gap between data science and engineering, to build and 
deploy data products. I'm not currently pursuing contract work. 

I've enjoyed building a unique combination of machine learning, deep learning, and software engineering skills. In my 
previous work at Capital One and startups, I've has built authorization fraud, insider threat, and legal discovery 
automation platforms. In each of these cases I've lead a team of data scientists and data engineers to enable and 
elevate our client's business workflow (and capture some amazing data).

When I'm not knee deep in a code base, I can be found traveling, sharing my collection of Japanese teas, and playing 
board games with my partner in Seattle. 

## Changelog

 - PR title (#PR number, or #Issue if no PR)
 - There's nothing here! (yet)

### Development

 - Updated README and setup.py links (No PR)

### 3.1.0

 - Add boolean datatype (#104)
 - Added Contributing.md section for new datatypes (#101)
 - Added datatypes to docs in index.rst (#101)
 - Modified documentation to automatically generate API docs (#101)
 

### 3.0.1

 - Changing CI to Circleci (#100)
 - Adding datatypes to CONTRIBUTING.md, adding CONTRIBUTING.md to docs (#96)
 - Adding docs badge (#95)
 - Adding support for unusual variable names / format keras names to be valid in name scope (#92)
 - Adding examples (#93)
 - Upgraded `requests` library to `requests==2.20.1`, based on security concern (#94)
 

### 3.0.0

Brand new release, with

Added

 - New `Datatype` interface, with easier to understand pipelines for each datatype
   - All existing datatypes (`Numerical`, `Categorical`, `Text` & `TimeSeries`) re-implmented in this new format
   - Support for custom data types generated by users
   - Duck-typing helper method (`keras_pandas/lib.check_valid_datatype()`) to confirm that a datatype has valid 
   signature
 - New testing, streamlined and standardized
 - Support for transforming unseen categorical levels, via the `UNK` token (experimental)
   
Modified

 - Updated `Automater` interface, which accepts a dictionary of data types
 - Heavily updated README
 - More consistent logging and data formatting for sample data sets

Removed

 - Removed examples, will be re-implemented in future release
 - All existing unittests
 - Bulk of new datatypes in `contributing.md`, will be re-added in future release
 
### 2.2.0

 - Add timeseries support (#78)
 - Add timeseries examples (#79)

### 2.1.0

 - Boolean support deprecated. Boolean (bool) data type can be treated as a special case of categorical data types

### 2.0.2

 - Remove a lot of the unnecessary dependencies (#75)
 - Update dependencies to contemporary versions (#74)
 
### 2.0.1

 - Fix issue w/ PyPi conflict

### 2.0.0

 - Adding CI/CD and PyPi links, and updating contact section w/ about the author (#70)
 - Major rewrite / update of examples (#72)
   - Fixes bug in embedding transformer. Embeddings will now be at least length 1. 
   - Add functionality to check if `resp_var` is in the list of user provided variables
   - Added better null filling w/ `CategoricalImputer`
   - Added filling unseen values w/ `CategoricalImputer`
   - Converted default transformer pipeline to use `copy.deepcopy` instead of `copy.copy`. This was a hotfix for a 
   previously unknown issue. 
   - Standardizing setting logging level, only in test base class and examples (when `__main__`)


### 1.3.5

 - Adding regression example w/ inverse_transformation (#64)
 - Fixing issue where web socket connections were being opened needlessly (#65)

### 1.3.4

 - Adding `Manifest.in`, with including files references in `setup.py` (#54) 

### 1.3.2

 - Fixed poorly written text embedding index unit test (#52)
 - Added license (#49)

### Earlier

 - Lots of things happened. Break things and move fast 
