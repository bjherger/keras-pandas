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