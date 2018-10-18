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

Adding support for new data types is designed to be (relatively) painless. A workflow for adding new data type (e.g. 
`VARTYPE`) includes:

 - Adding a tests to `test/`
 - Modifying `Automater.__init__`
   - Include new data type in `Automater.__init__`'s parameters (e.g. `VARTYPE_vars=list()`)
   - Add the new variable to `self._variable_type_dict` (e.g. `self._variable_type_dict['VARTYPE_vars'] = VARTYPE`)
 - Modifying `constants.py` to add input support
   - Updating `default_sklearn_mapper_pipelines` to include the SKLearn transformations to perform for this data type 
   (e.g. `'VARTYPE_vars': [LabelEncoder()]`)
   - Creating an input nub handler function (e.g. `def input_nub_VARTYPE_handler(variable, input_dataframe)`)
   - Adding the input nub handler to default_input_nub_type_handlers (e.g. adding 
   `'VARTYPE_vars': input_nub_VARTYPE_handler`)
 - Modifying `constants.py` to add output support (optional)
   - Updating `default_suggested_losses` to include a suggested loss (e.g. `'VARTYPE_vars': losses.mean_squared_error`)
 - Modifying `Automater.py` to add output support (optional)
   - Updating `_create_output_nub` to create an output layer
   - Updating `inverse_transform_output` to inverse transform Keras outputs