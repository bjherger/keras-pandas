# `keras-pandas` examples

A few examples, highlighting the `keras-pandas` interface. All examples should be based on `example_interface.py`

## `example_interface.py`

Problem type: TODO

Input data types: TODO

Description: TODO


## `lending_club_predict_loan_status.py`

Problem type: Regression

Input data types: numerical, categorical, text

Description: Predicting a user's dti (debt to income ratio) 

## `instanbul_predict_ise.py`

Problem type: Regression

Input data types: numerical, timeseries

Description: Using historical stock data and values for other stocks, to predict the value of a single stock (`ise`).
 This script highlights the expected format for timeseries data. Timeseries input data should be an array of values 
 (e.g. `[val1, val2, val3]`)


