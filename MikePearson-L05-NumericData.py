#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 14:47:36 2018

@author: mutecypher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
Wines= pd.read_csv(url, header=None)

## See what the dimensions of Wines are

print(Wines.shape)

## Get the column headers from the data description and add them in

Wines.columns = ["Class", "Alcohol","Malic Acid","Ash","Alcalinity","Magnesium","Total phenols","Flavanoids","nonflvanoids","procnthocyanins","Color Intensity","Hue", "OD280", "Proline"]

## See what the data looks like in Wines

print(Wines.loc[:,"Alcohol"].dtypes)



## Since its a float64 data type, there must not be any "?" or "NaN"
## Just for the sake of this exercise, introduce some random "?" in the "Alcohol" column
## Create a data frame with 7 random numbers between 0 and 89 to use as row numbers to insert "?"

randq = np.random.randint(0,89, 7)

## Now insert "?" in random locations just to remove them later, check to see

Wines.loc[randq,"Alcohol"] = "?"

print(Wines.loc[:,"Alcohol"].unique())

## Do the same sort of thing to create 5 random float("NaN")s in rows 90 to 178 of the column "Alcohol"

randnan = np.random.randint(90, 177, 5)

Wines.loc[randnan, "Alcohol"]= float("NaN")


## For the final bit of fun, introduce an outlier at a random location in "Alcohol"

Wines.loc[np.random.randint(90, 177,1) , "Alcohol"] = "31"
Wines.loc[np.random.randint(0,89,1), "Alcohol"] = "4"

## Now that we have a data set with missing values, let's clean it up 
## First get rid of the "?"

Wines.loc[:,"Alcohol"]= Wines.loc[:,"Alcohol"].replace(to_replace = "?", value = float("NaN"))

## Now convert the the column to numberic

Wines.loc[:,"Alcohol"] = pd.to_numeric(Wines.loc[:,"Alcohol"], errors = 'coerce')

## see if that cleaned things up

print(Wines.loc[:,"Alcohol"].unique())

## All good
## Now find the median of the data without the nan's

Alc_med = np.nanmedian(Wines.loc[:,"Alcohol"])


## Now replace the nan's with the median

Leave_out = np.isnan(Wines.loc[:,"Alcohol"])
Wines.loc[Leave_out,"Alcohol"]= Alc_med


## And for the last little bit, remove any outliers - using our standard definition of two
## standard deviations away from the mean

min_Alc = np.mean(Wines.loc[:,"Alcohol"]) - 2*np.std(Wines.loc[:,"Alcohol"])
max_Alc = np.mean(Wines.loc[:,"Alcohol"]) + 2*np.std(Wines.loc[:,"Alcohol"])

##  Flag the entries that are outliers
FlagOuts = (Wines.loc[:,"Alcohol"] < min_Alc) | (Wines.loc[:,"Alcohol"] > max_Alc)

## Replace the outlier values with the median

Wines.loc[FlagOuts, "Alcohol"]= Alc_med


## Now let's do the final histogram, change the range to account for removing the outliers

low2 = np.floor(min(Wines.loc[:,"Alcohol"]))
high2 = np.ceil(max(Wines.loc[:,"Alcohol"]))+ 0.5

## And plot

without_outliers = plt.hist(Wines.loc[:,"Alcohol"], range = (low2, high2))
plt.title("Alcohol with Outliers replaced by medians")
print(without_outliers)

###  I loaded the Wines data set from UCI Machine Learning
### I gave the columns their proper headers
## The data was all reasonable, but for the sake of the exercise
## I randomly placed "?" and nan values into the Alcohol column
## I also placed a couple of outlier values into the column
## I then replaced the "?" values with float("Nan")
## And then coerced the "Alcohol" column back to float64
## The median was calculated for "Alcohol" without the nan values
## The nan values were replaced with the median
## The outliers were also replaced with the median value
## And finally a histogram was made of the alcohol values