#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 08:38:34 2018

@author: mutecypher
"""

#### Begin with the lpandas library, shorten the name
import pandas as pd
import requests as rq

## now let's set the url of the UCI Machine Learning website with data on wine

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wines_desc = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names"

## A quick peak at the file from the website confirms that it is a csv file
## so we will pull it down from the website with a read_csv command

wines_df = pd.read_csv(url, header = None)

## Now let's print the first 5 rows of that data

print(wines_df[:5])

## let's get the attribute names from the names file
wines_txt = rq.get(wines_desc).text

## for grins, let's print out the list of attributes
## I know this is a clunky way to do it, I will get better at this...

print(wines_txt[2307:2620])



### let's assign column names to our data using the description in the print

wine_columns= ["Class", "Alcohol", "Malic acid","Ash", "Aclalinity of ash", "Magnesium","Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyananins","Color Intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
wines_df.columns = wine_columns

## now let's look again at our data frame
print(wines_df[:5])


##  I imported the pandas and requests libraries
## I found the url for wine data at the machine learning data depository - a csv file
## I found the url for the description of the data - a text file
## I printed out the first 5 rows of the wine data
## I printed the portion of the description file that gave the attributes
## I assigned the columns of my data frame the proper names 
## I once again printed out the first 5 rows of data, now with proper attribute names