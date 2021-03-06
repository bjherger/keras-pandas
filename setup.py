from collections import OrderedDict

from setuptools import setup, find_packages

__version__ = '3.1.0'

# Add README as long description
with open("README.md", "r") as fh:
    long_description = fh.read()

# Parse requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='keras-pandas',
    description='Easy and rapid deep learning',
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    url='https://github.com/bjherger/keras-pandas',
    author='Brendan Herger',
    author_email='13herger@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=required,
    project_urls=OrderedDict((
        ('Code', 'https://github.com/bjherger/keras-pandas'),
        ('Documentation', 'http://keras-pandas.readthedocs.io/en/latest/intro.html'),
        ('PyPi', 'https://pypi.org/project/keras-pandas/'),
        ('Issue tracker', 'https://github.com/bjherger/keras-pandas/issues'),
        ('CI/CD', 'https://circleci.com/gh/bjherger/keras-pandas/tree/master'),
        ('Author\'s website', 'https://www.hergertarian.com/')
    )),
)
