#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 18:13:32 2018

@author: mutecypher
"""
## import our pal numpy and give him a shortened name for easy of typing

import numpy as np

## create an array with missing values and an outlier



x = np.array([2, 128, 5 , 6, 13, 8, 1, 23, float("nan"), 8, 15, " ", 12, "?", -80])

## clean the data of non-numeric values "?" and " "


FlagNonNums = (x != "?") & (x != " ")
x =x[FlagNonNums]

## What type of data is the array now?

x.dtype.name

## That array has been changed to strings, deal with it as a string then change
## Remove the nan string

NumFlags = (x != "nan")

x = x[NumFlags]

## Now convert the array to integer, instead of a string

x=np.asfarray(x,int)


## now let's look at the distribution of data


LimitHi = np.mean(x) + 2*np.std(x)
LimitLo = np.mean(x) - 2*np.std(x)

## See what values fall within this range


y = x[(x <= LimitHi) & (x >= LimitLo)]

## View these in the Variable Explorer,since it's a small dataset

## Make a flag for the outliers

FlagOuts = (x < LimitLo) | (x > LimitHi)

## Now replace the outliers with the median value of the array 

x[FlagOuts] = np.median(x)

x

##  The array has been cleaned  of non-mumeric inputs like "?" and " "
## then cleaned of NaN's
##Then the array was converted to integer, after having been a string array
## The outliers were identified using the 'mean plus or minus two standard deviations" heuristic
## then the outliers were replaced with the median of the outlier-free data