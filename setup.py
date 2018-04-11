import re

from setuptools import setup

# Set up requirements

with open('requirements.txt') as f:
    required = f.read().splitlines()
    required = required[2:]
    required = map(lambda x: x.strip(), required)
    required = map(lambda x: re.sub(r'\s+', '==', x), required)

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


