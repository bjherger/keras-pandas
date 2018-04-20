import re

from setuptools import setup

# Set up requirements

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='auto_dl',
    description='test',
    version='0.1.1',  # Update the version number for new releases
    url='https://github.com/bjherger/auto_dl',
    author='Brendan Herger',
    author_email='13herger@gmail.com',
    license='MIT',
    packages=['auto_dl'],
    install_requires=required,
    zip_safe=False
)


