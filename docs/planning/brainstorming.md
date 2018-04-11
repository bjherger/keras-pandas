# Elevator pitch

# Audience

 - Semi-technical
 - ML background, new to deep learning
 - Familiarity with Python data ecosystem (Pandas, SKLearn, numpy)
 - Interested in getting started in deep learning
 - Interested in quickly prototyping models

# Goals

 - Easy to use interface
 - Sensible default actions
 - Low barrier to entry to create Keras input / output layers

# Requirements

## Backlog

 - Numerical inputs: Null handling, Z score normalizaiton
 - Categorical inputs: Create embedding, handle unseen levels
 - Boolean inputs: Handle appropriately
 - Datetime: Extract categorical fields, treat as epoch time if possible. 
 - Test run: train on random sample of data
 - Convenient interface
 - Logging
 - Unit tests
 - Appropriate exceptions
 - Pip installable 

## Prioritized backlog

 - Unit tests
 - Logging
 - Numerical inputs: Null handling, Z score normalizaiton
 - Categorical inputs: Create embedding, handle unseen levels
 - Boolean inputs: Handle appropriately
 - Datetime: Extract categorical fields, treat as epoch time if possible. 
 - Appropriate exceptions
 - Pip installable 

## POC items

 - Interface: Need to determine options (SKLearn transformer, custom interface, etc)
 - Interface: Need to outline functionality
 - Boolean: Need to determine if it'll be handled as numerical or categorical
 - Pip installable: Need to determine level of effort