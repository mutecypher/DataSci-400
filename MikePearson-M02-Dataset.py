#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 09:09:29 2018

@author: mutecypher
"""

import pandas as pd
import numpy as np


## I will use the Heart Disease data set for this assignment

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"

hearty = pd.read_csv(url, header=None)

## Having read the heart-disease.names data file, I assign names to the columns

hearty.columns= ["age", "sex", "cp", "trestbps", "chol", "fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]
print(hearty.head())

## What does the data look like?

print(hearty.dtypes)

## The trestbps column should be  numeric, but it's an object, let's see if there are missing values

print(hearty.loc[:, "trestbps"].unique())

### yup, let's replace the missing values with floating NaNs

hearty.loc[:,"trestbps"]= hearty.loc[:,"trestbps"].replace(to_replace = "?", value = float("NaN"))

## Now let's change to numeric

hearty.loc[:,"trestbps"] = pd.to_numeric(hearty.loc[:,"trestbps"], errors = 'coerce')


## Now let's find the median of the nan-excluded data, and impute the median to the nans

NANbps = np.nanmedian(hearty.loc[:,"trestbps"])


## Now replace the nan's with the median

bps_out = np.isnan(hearty.loc[:,"trestbps"])
hearty.loc[bps_out,"trestbps"]= NANbps


## Change the sex column to  new columns of "Male" and "Female"
hearty.loc[:, "Female"] = (hearty.loc[:, "sex"] == 0).astype(int)
hearty.loc[:, "Male"] = (hearty.loc[:, "sex"] == 1 ).astype(int)

## Let's look at the "chol" cholesterol column, it should be integer

##print(hearty.loc[:,"chol"].unique())
## Yup, a missing value - Let replace the "?" with float(NaN) and then 
## the mean value

hearty.loc[:,"chol"]= hearty.loc[:,"chol"].replace(to_replace = "?", value = float("NaN"))
hearty.loc[:,"chol"] = pd.to_numeric(hearty.loc[:,"chol"], errors = 'coerce')
NANchol = np.nanmean(hearty.loc[:,"chol"])
chol_out = np.isnan(hearty.loc[:,"chol"])
hearty.loc[chol_out,"chol"]= NANchol

## Now for the fbs column "fasting blood sugar"

##print(hearty.loc[:,"fbs"].unique())

## some values are missing, follow the standard procedure to replace "?" with float(NaN)

hearty.loc[:,"fbs"]= hearty.loc[:,"fbs"].replace(to_replace = "?", value = float("NaN"))
hearty.loc[:,"fbs"] = pd.to_numeric(hearty.loc[:,"fbs"], errors = 'coerce')

## What is the most common value =, either 1 (above 120) or 0 (below 120)
print(hearty.loc[:,"fbs"].value_counts())

## The most common (by 13x) is below 120, so impute that to all the missing values

chol_out = np.isnan(hearty.loc[:,"fbs"])
hearty.loc[chol_out,"fbs"] = 0

## change these to columns of "above 120 mg/dl" "below 12)0 mg/dl"

hearty.loc[:,"above 120 mg/dl"] = (hearty.loc[:,"fbs"] == 1).astype(int)
hearty.loc[:,"below 120 mg/dl"] = (hearty.loc[:,"fbs"] == 0).astype(int)

## Now look at resting ecg

##print(hearty.loc[:,"restecg"].unique())

print(hearty.loc[:,"restecg"].value_counts())

## convert to numeric, replace the sole "?" with "0" - the most common value

hearty.loc[:,"restecg"]= hearty.loc[:,"restecg"].replace(to_replace = "?", value = float("NaN"))
hearty.loc[:,"restecg"] = pd.to_numeric(hearty.loc[:,"restecg"], errors = 'coerce')

restecg_out = np.isnan(hearty.loc[:,"restecg"])
hearty.loc[restecg_out, "restecg"]= 0

## now create columns of the values "normal", "ST-T abnormal"," left v htrophy"

hearty.loc[:,"normal ecg"] = (hearty.loc[:,"restecg"] == 0).astype(int)
hearty.loc[:,"ST-T abnormal"] = (hearty.loc[:,"restecg"] == 1).astype(int)
hearty.loc[:,"left v htrophy"] = (hearty.loc[:,"restecg"] == 2).astype(int)


## Now look at "thalach" the maximum heart rate achieved

##print(hearty.loc[:,"thalach"].unique())


## yup, more missing data. repeat the standard steps and replace missing with the mean

hearty.loc[:,"thalach"]= hearty.loc[:,"thalach"].replace(to_replace = "?", value = float("NaN"))
hearty.loc[:,"thalach"] = pd.to_numeric(hearty.loc[:,"thalach"], errors = 'coerce')
NaN_thalach = np.nanmean(hearty.loc[:,"thalach"])
thalach_out = np.isnan(hearty.loc[:,"thalach"])
hearty.loc[thalach_out, "thalach"] = NaN_thalach

#3 Okay, let's look at exang

##print(hearty.loc[:,"exang"].unique())

hearty.loc[:,"exang"]= hearty.loc[:,"exang"].replace(to_replace = "?", value = float("NaN"))
hearty.loc[:,"exang"] = pd.to_numeric(hearty.loc[:,"exang"], errors = 'coerce')

## find the most common value for imputation

print(hearty.loc[:,"exang"].value_counts())

## it's 0, replace NaN with 0

exang_out = np.isnan(hearty.loc[:,"exang"])
hearty.loc[restecg_out, "exang"] = 0

## Turn these values into columns "exercise induced" or "not exercise induced"

hearty.loc[:,"xrcise induced"] = (hearty.loc[:,"exang"] == 1).astype(int)
hearty.loc[:,"not xrcise induced"] = (hearty.loc[:,"exang"] == 0).astype(int)


## looking at the slope column now

##print(hearty.loc[:,"slope"].unique())

## replace "?" with floatNaN

hearty.loc[:,"slope"]= hearty.loc[:,"slope"].replace(to_replace = "?", value = float("NaN"))
hearty.loc[:,"slope"] = pd.to_numeric(hearty.loc[:,"slope"], errors = 'coerce')

## what's the most common value for imputation

print(hearty.loc[:,"slope"].value_counts())

## it's the number 2, replace NaN with 2

slope_out = np.isnan(hearty.loc[:,"slope"])
hearty.loc[slope_out, "slope"]= 2

## Now create columns for "upslope", "flat", and "downslope"

hearty.loc[:,"upslope"] = (hearty.loc[:,"slope"] == 1).astype(int)
hearty.loc[:,"flat"] = (hearty.loc[:,"slope"] == 2).astype(int)
hearty.loc[:,"downslope"] = (hearty.loc[:,"slope"] == 3).astype(int)

## Now look at the number of vessels colored by fluorscopy

##print(hearty.loc[:,"ca"].unique())

## looks like all values are either 0 or missing, no useful data,remove column



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
hearty = hearty.drop("sex", axis = 1)
hearty = hearty.drop("fbs", axis = 1)
hearty = hearty.drop("restecg", axis = 1)
hearty = hearty.drop("cp", axis = 1)
hearty = hearty.drop("ca", axis = 1)
hearty = hearty.drop("thal", axis = 1)
hearty = hearty.drop("exang", axis = 1)

##  Time to bin the age data

NB = 5

bounds = np.linspace(np.min(hearty.loc[:,"age"]) - 1, np.max(hearty.loc[:,"age"]) + 1, NB + 1)

## define a function
def bin(x, b): # x = data array, b = boundaries array
    nb = len(b)
    N = len(x)
    y = np.empty(N, int) # empty integer array to store the bin numbers (output)
    
    for i in range(1, nb): # repeat for each pair of bin boundaries
        y[(x >= bounds[i-1])&(x < bounds[i])] = i
    
    y[x == bounds[-1]] = nb - 1 # ensure that the borderline cases are also binned appropriately
    return y

###use the function
    
hearty.loc[:,"agebins"] = bin(hearty.loc[:,"age"], bounds)

##

hearty.loc[:, "28 to 34"] = (hearty.loc[:,"agebins"] == 1).astype(int)
hearty.loc[:, "35 to 42"] = (hearty.loc[:,"agebins"] == 2).astype(int)
hearty.loc[:, "43 to 50"] = (hearty.loc[:,"agebins"] == 3).astype(int)
hearty.loc[:, "51 to 58"] = (hearty.loc[:,"agebins"] == 4).astype(int)
hearty.loc[:, "59 to 66"] = (hearty.loc[:,"agebins"] == 5).astype(int)


## time to normalize some data
hearty = hearty.drop("age", axis = 1)
hearty = hearty.drop("agebins", axis = 1)


## sort through "oldpeak" for outliers

oldpeak_hi = np.mean(hearty.loc[:,"oldpeak"]) + 2*np.std(hearty.loc[:,"oldpeak"])
oldpeak_lo = np.mean(hearty.loc[:,"oldpeak"]) - 2*np.std(hearty.loc[:,"oldpeak"])

badpeak = (hearty.loc[:,"oldpeak"] < oldpeak_lo) | (hearty.loc[:,"oldpeak"] > oldpeak_hi)

oldpeak_mean = np.mean(hearty.loc[:,"oldpeak"])

hearty.loc[badpeak,"oldpeak"] = oldpeak_mean


## normalize the "oldpeak" data with minmax normalization

oldpeak_min = np.min(hearty.loc[:,"oldpeak"])
oldpeak_range = np.max(hearty.loc[:,"oldpeak"]) - np.min(hearty.loc[:,"oldpeak"])

hearty.loc[:,"oldpeak normed"] = (hearty.loc[:,"oldpeak"] - oldpeak_min)/oldpeak_range

hearty = hearty.drop("oldpeak", axis = 1)
                     
## now norm trestbps - but first, let's kill the outliers

trestbps_hi = np.mean(hearty.loc[:,"trestbps"]) + 2*np.std(hearty.loc[:,"trestbps"])
trestbps_lo = np.mean(hearty.loc[:,"trestbps"]) - 2*np.std(hearty.loc[:,"trestbps"])

badbps = (hearty.loc[:,"trestbps"] < trestbps_lo) | (hearty.loc[:,"trestbps"] > trestbps_hi)

trestbps_mean = np.mean(hearty.loc[:,"trestbps"])

hearty.loc[badbps,"trestbps"] = trestbps_mean

trestbps_min = np.min(hearty.loc[:,"trestbps"])
trestbps_range = np.max(hearty.loc[:,"trestbps"]) - np.min(hearty.loc[:,"trestbps"])

hearty.loc[:,"restbps normed"] = (hearty.loc[:,"trestbps"] - trestbps_min)/trestbps_range

hearty = hearty.drop("trestbps", axis = 1)

##  Look at cholesterol and follow the same steps

chol_hi = np.mean(hearty.loc[:,"chol"]) + 2*np.std(hearty.loc[:,"chol"])
chol_lo = np.mean(hearty.loc[:,"chol"]) - 2*np.std(hearty.loc[:,"chol"])

badchol = (hearty.loc[:,"chol"] < chol_lo) | (hearty.loc[:,"chol"] > chol_hi)

chol_mean = np.mean(hearty.loc[:,"chol"])

hearty.loc[badchol,"chol"] = chol_mean

chol_min = np.min(hearty.loc[:,"chol"])
chol_range = np.max(hearty.loc[:,"chol"]) - np.min(hearty.loc[:,"chol"])

hearty.loc[:,"chol normed"] = (hearty.loc[:,"chol"] - chol_min)/chol_range

hearty = hearty.drop("chol", axis = 1)

### and now for thalach

thalach_hi = np.mean(hearty.loc[:,"thalach"]) + 2*np.std(hearty.loc[:,"thalach"])
thalach_lo = np.mean(hearty.loc[:,"thalach"]) - 2*np.std(hearty.loc[:,"thalach"])

badthalach = (hearty.loc[:,"thalach"] < thalach_lo) | (hearty.loc[:,"thalach"] > thalach_hi)

thalach_mean = np.mean(hearty.loc[:,"thalach"])

hearty.loc[badthalach,"thalach"] = thalach_mean

thalach_min = np.min(hearty.loc[:,"thalach"])
thalach_range = np.max(hearty.loc[:,"thalach"]) - np.min(hearty.loc[:,"thalach"])

hearty.loc[:,"thalach normed"] = (hearty.loc[:,"thalach"] - thalach_min)/thalach_range

hearty = hearty.drop("thalach", axis = 1)
hearty = hearty.drop("Male", axis =1)
hearty = hearty.drop("below 120 mg/dl", axis = 1)
hearty = hearty.drop("normal ecg", axis = 1)
hearty = hearty.drop("not xrcise induced", axis = 1)
hearty = hearty.drop("asymptomatic", axis = 1)

hearty.to_csv('/Users/mutecypher/Documents/UW work/GitHub/DataSci-400/MikePearson-M02-Dataset.csv', sep =',', header = ["num", "Female", "above 120 mg/dl",
                                                "ST-T abnormal"," left v htrophy","xrcise induced","upslope","flat","angina","non-angina",
                                                "28 to 34","35 to 42","43 to 50","51 to 58","59 to 66","oldpeak normed",
                                                "restbps normed","chol normed","thalach normed"])

