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
    long_description_content_type='text/markdown',
    version='1.1.0',
    url='https://github.com/bjherger/keras-pandas',
    author='Brendan Herger',
    author_email='13herger@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=required,
    zip_safe=False
)


