from collections import OrderedDict

from setuptools import setup, find_packages

# Set up requirements

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='keras-pandas',
    description='Easy and rapid deep learning',
    long_description=long_description,
    long_description_content_type="text/markdown",
    version='1.1.1',
    url='https://github.com/bjherger/keras-pandas',
    author='Brendan Herger',
    author_email='13herger@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=required,
    project_urls=OrderedDict((
        ('Documentation', 'https://github.com/bjherger/keras-pandas'),
        ('Code', 'https://github.com/bjherger/keras-pandas'),
        ('Issue tracker', 'https://github.com/bjherger/keras-pandas/issues'),
    )),
)


