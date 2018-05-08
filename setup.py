from setuptools import setup

# Set up requirements

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='keras-pandas',
    description='test',
    version='0.2.0',  # Update the version number for new releases
    url='https://github.com/bjherger/keras-pandas',
    author='Brendan Herger',
    author_email='13herger@gmail.com',
    license='MIT',
    packages=['keras_pandas'],
    install_requires=required,
    zip_safe=False
)


