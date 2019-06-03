# Spec

#### Problem def
- ML models are not scalable in jupyter for large ML projects. Need a new model of working.

#### Requirements
- a repo which provides clear and reproducible output for SGH classification of overcrowding model
- possibility of reusing function that are written for new ML projects.

#### Users
- BK/TM

## Use Cases

## Features

|Use|Description|Priority|Done|
|---|-----------|--------|----|
|**Data Prep**||||
|Load Raw Feature Data|ED files, IP files.||
|Load Target Data|Load target(daily count of occupancy).|||
|Calc target outliers|Find classes for each day: based on simple method.||
||||
|**Feature Selection**|||
|Morning features|Calculates current status of ED/IP at a particular hour for each day. Provide a list of datetimes at set hour each day. Returns daily df.||
|Daily features|Summary of activity over an entire day. IP and ED. Returns daily df.||
|Lagging daily features|Add new features based on previous lagged features. Provide feature and list of which lags required. Adds new columns to df.||
|Derivatives features|Create new feature using exisiting feature, smoothing, calc derivative/diff. Adds new columns to df.||
||||
|**Modelling**|||
|Split Test/Train|Take final year of data and put to one side.||
|Run several basic models|Run models with CV and record results in table.||
|Grid Search|Select a model and put into grid seach function. Add results to the basic model table/new table.||
||||
|**Eval/vis**|||
|Eval model on test set function|Given a model that has been fitted and test data, calculate metrics on test set and plot PR and ROC curves.||y|
||||