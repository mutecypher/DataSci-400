#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 08:22:34 2018

@author: mutecypher
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## I will use the Heart Disease data set for this assignment

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"

hearty = pd.read_csv(url, header = None)

## Having read the heart-disease.names data file, I assign names to the columns

hearty.columns= ["age", "sex", "cp", "trestbps", "chol", "fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]
print(hearty.head())

## What does the data look like?

print(hearty.dtypes)

## The trestbps column should be  numeric, but it's an object, let's see if there are missing values

print(hearty.loc[:,"trestbps"].unique())

### yup, let's replace the missing values with floating NaNs

hearty.loc[:,"trestbps"]= hearty.loc[:,"trestbps"].replace(to_replace = "?", value = float("NaN"))

## Now let's change to numeric

hearty.loc[:,"trestbps"] = pd.to_numeric(hearty.loc[:,"trestbps"], errors = 'coerce')


## Now let's find the median of the nan-excluded data, and impute the median to the nans

NANbps = np.nanmedian(hearty.loc[:,"trestbps"])


## Now replace the nan's with the median

Leave_out = np.isnan(hearty.loc[:,"trestbps"])
hearty.loc[Leave_out,"trestbps"]= NANbps


## Change the class categories to 
hearty.loc[:, "upslope"] = (hearty.loc[:, "slope"] == "1").astype(int)
hearty.loc[:, "flat"] = (hearty.loc[:, "slope"] == "2").astype(int)
hearty.loc[:, "downslope"] = (hearty.loc[:, "slope"] == "3").astype(int)


### And let's look at the cp column  - type of chest pain
## can consolidate typical angina, and atypical angina# let's just call them 'angina'

hearty.loc[:,"cp"] = hearty.loc[:,"cp"].replace(to_replace = 1, value = "angina")
hearty.loc[:,"cp"] = hearty.loc[:,"cp"].replace(to_replace = 2, value = "angina")
hearty.loc[:,"cp"] = hearty.loc[:,"cp"].replace(to_replace = 3, value = "non-angina")
hearty.loc[:,"cp"] = hearty.loc[:,"cp"].replace(to_replace = 2, value = "asymptomatic")


## Now change the class categories to new columns of "angina", "non-angina", "asymptomatic"


hearty.loc[:, "angina"] = (hearty.loc[:, "cp"] == "angina").astype(int)
hearty.loc[:, "non-angina"] = (hearty.loc[:, "cp"] == "non-angina").astype(int)
hearty.loc[:, "asymptomatic"] = (hearty.loc[:, "cp"] == "asymptomatic").astype(int)


## now plt the admitted rest bps for angina

## No longer need the "slope" column, nor do we need the "downslope column" since it can be 
## derived from where "upslope" and "flat" = 0

hearty = hearty.drop("slope", axis = 1)
hearty = hearty.drop("downslope", axis = 1)

## also, don't need the cp column, nor the 'asymptomatic" column, since it can be derived
## from "angina" and "non-angina" both equaling 0

hearty = hearty.drop("cp", axis = 1)
hearty = hearty.drop("asymptomatic", axis = 1)


angie = hearty.loc[:,"angina"]== 1
no_angie = hearty.loc[:,"non-angina"] == 1

bob = plt.hist(hearty.loc[angie,"trestbps"], color = "skyblue", label = "with angina")
tod = plt.hist(hearty.loc[no_angie,"trestbps"], color = "crimson", label = "no angina")
tam = plt.legend()
title = plt.suptitle("Rest BPS Upon Admission")
print(bob, tod, tam, title)
### I imported the needed packages: numpy, pandas, and matplotlib.pyplot
## I read in the Heart Disease database from the UCI  Machine Learning depoisitory
## After examining the data, I assigned the appropriate column names
## Then I chose the resting blood pressure (bps) as a set of data to examing
## I found missing values in the bps and then imputed the median to those missing values
## Just for the exercise, I decoded the category data for slope - the slope of the peak exercise 
## segment, and the "cp" (chest pain) columns
## I combined the categories of "typical angina" and "atypical angina" in the cp (chest pain)
## column, then consildated categories
## I removed the obsolete columns of slope, and downslope, and of cp and asymptomatic
## I then plotted a histogram of the rest blood pressure upon admittance of
## the patients with angina (sky blue) and without angina(red)