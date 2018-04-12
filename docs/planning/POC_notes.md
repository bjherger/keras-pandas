# Backlog 

 - Interface: Need to determine options (SKLearn transformer, custom interface, etc)
 - Interface: Need to outline functionality
 - Boolean: Need to determine if it'll be handled as numerical or categorical
 - Pip installable: Need to determine level of effort

# Interface: Need to determine options (SKLearn transformer, custom interface, etc)

Options: 

 - SKLearn transformer
 - Pandas-SKLearn style module
 - Custom module

Requirements:

 - Be able to expand one column to many columns (datetime)
 - Abilityt o use SKLearn transformers

## SKLearn transformer

 - Inputs would have to be Numpy arrays
 - Inputs can be Numpy arrays
 - Tighter integration to SKLearn infrastructure

## Pandas-SKLearn style module

 - Can have pandas dataframe inputs
 - Can have multipel column inputs
 - Can apply multiple transformations to the same column (many to one relationship)
 - Can not apply transformations to some columns
 - Can use default transformer

## Custom module

 - Can sit on top of Pandas-SKLearn
 - Can mimic SKLearn `fit` and `transform` interface

#  Need to determine if it'll be handled as numerical or categorical

## Numerical 

 - Less compute time
 - Reduced complexity

## Categorical

 - Embedding representing different values

# Pip installable: Need to determine level of effort

## Lit review

 - PMOTW: Setuptools
 - Common library: setuptools
 - PMOTW: distutils
 - Common library: distutils
 - Blog review

## PMOTW: Setuptools

 - Unavailable

## [Common library: setuptools](https://setuptools.readthedocs.io/en/latest/)

 - Designed to facilitate packaging Python projects
 - Enhancement to distutils

Highlights

 - Dependency resolution & downloading
 - Create eggs
 - Automatically create wrapper scripts & .exe files
 - [Decent getting started guide](https://setuptools.readthedocs.io/en/latest/setuptools.html#basic-use)

Superset of of distutils

## PMOTW: distutils

 - Unavailable

## [Common library: distutils](https://docs.python.org/2/distutils/)

[Intro](https://docs.python.org/2/distutils/introduction.html)

 - Setup script
 - Source distribution
 - Binary distributions

Setup script

 - Handles packaging
 - Not aware of package managers

[PyPi](https://docs.python.org/2/distutils/packageindex.html#pypi-overview)

 - Registering
 - Upload

## Blogs

 - [Marthall](https://marthall.github.io/blog/how-to-package-a-python-app/)
 - [so](https://stackoverflow.com/questions/9411494/how-do-i-create-a-pip-installable-project)

## [Marthall](https://marthall.github.io/blog/how-to-package-a-python-app/)

 - Strong, convenient walk through

## [so](https://stackoverflow.com/questions/9411494/how-do-i-create-a-pip-installable-project)

 - Rambling

## [Scott Torborg](http://python-packaging.readthedocs.io/en/latest/)

- Strong advanced discussion
- How to declare dependencies

## [PyPI docs](https://packaging.python.org/tutorials/distributing-packages/#uploading-your-project-to-pypi)

## [twine quickstart](https://packaging.python.org/tutorials/distributing-packages/)

# Decisions

 - Interface: Will use custom interface, similar to SKLearn, with Pandas-SKLearn under the hood.
 - Pip installable: Will move forward w/ setuptools