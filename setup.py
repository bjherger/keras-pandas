from collections import OrderedDict

from setuptools import setup, find_packages

__version__ = '2.1.0'

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
        ('Documentation', 'http://keras-pandas.readthedocs.io/en/latest/intro.html'),
        ('Code', 'https://github.com/bjherger/keras-pandas'),
        ('Issue tracker', 'https://github.com/bjherger/keras-pandas/issues'),
        ('PyPi', 'https://pypi.org/project/keras-pandas/'),
        ('CI/CD', 'https://travis-ci.org/bjherger/keras-pandas/builds'),
        ('Author\'s website', 'https://www.hergertarian.com/')
    )),
)


