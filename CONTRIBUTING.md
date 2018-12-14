# Contributing

If you're interested in helping out, all open tasks are listed the GitHub Issues tab. The issues tagged with 
`first issue` are a good place to start if your new to the project or new to open source projects. 

If you're interested in a new major feature, please feel free to reach out to me

## Bug reports

The best bug reports are Pull Requests. The second best bug reports are new issues on this repo.

## Test

This framework uses `unittest` for unit testing. Tests can be run by calling:

```bash
cd tests/

python -m unittest discover -s . -t .
```
## Style guide

This codebase should follow [Google's Python Style Guide](https://google.github.io/styleguide/pyguide.html). 

## Changelog

If you've changed any code, update the changelog on `README.md`

## Generating documentation

This codebase uses [sphinx](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)'s 
[autodoc](http://www.sphinx-doc.org/en/master/ext/autodoc.html) feature. To generate new documentation, to reflect 
updated documentation, run:

```bash
cd docs

make html

```  

## Adding new data types


If there's a specific datatype you'd like to use that's not built in (such as images, videos, or geospatial), you can 
include it by using `Automater`'s `datatype_handlers` parameter. 

A template datatype can be found in `keras_pandas/data_types/Abstract.py`. Filling out this template will yield a new
 datatype handler. If you're happy with your work and want to share your new datatype handler, create a PR.
 
To create add a new datatype:

 - Create a new `.py` file in `keras_pandas/data_types`, based on `keras_pandas/data_types/Abstract.py` (and perhaps 
 referencing `keras_pandas/data_types/Numerical.py`)
 - Fill out your new datatype's `.py` file
 - Create a new test class for your new datatype (perhaps based on `tests/testDatatypeTemplate.py` and / or 
 `tests/testNumerical.py`) 
 - Add the new datatype to `keras_pandas/Automater.datatype_handlers`, in `keras_pandas/Automater.__init__()`
 - Add the new datatype to `docs/index.rst`, in `autosummary list` 

## Adding new examples

To contribute a new example

 - Add data loader method to `keras_pandas/lib.py` (perhaps in the style of `load_titanic()`)
 - Add a new `.py` file under `examples` (perhaps by copying and pasting `example_interface.py`)
 - Implement the required steps
 - Add the new file to `tests/testExamples.py`
 - Add the new example to `examples/README.md`
