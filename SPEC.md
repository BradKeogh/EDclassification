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
|**Data Prep**|Load raw data files|||
|Load Feature Data|- ED files, IP files||
|Load Target Data| Load target(daily count of occupancy)|||
|Calc target outliers|Find classes for each day: based on simple method||
||||
|**Feature Selection**|||
|Morning features|||
|Daily features|||
|Lagging daily features|||
||||
|**Modelling**|||
|Split Test/Train|||
|Run several basic models|run models with CV and record results in table.||
|Grid Search|Select a model and put into grid seach function||
||||
|**Eval/vis**|||
|Eval model on test set function| | ||
||||