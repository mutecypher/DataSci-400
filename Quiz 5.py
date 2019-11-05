#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 10:17:08 2018

@author: mutecypher
"""

import pandas as pd
import matplotlib.pyplot as plt
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
Adult = pd.read_csv(url, header=None)
# http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
Adult_columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
Adult.columns = Adult_columns + ["Income"]
Adult.dtypes

import pandas as pd
Houses = pd.DataFrame()
Houses.loc[:,"NumOfRooms"]  = [2, 4, "<2", 2, 3]
OneOrNone = [False, False, True, False, False]
OneOrNone = Houses.loc[:, "NumOfRooms"] == "<2"
Houses.loc[OneOrNone,"NumOfRooms"] = "1"


import pandas as pd
Houses = pd.DataFrame()
Houses.loc[:,"NumOfRooms"]  = [2, 4, "1", 2, "3"]
# The Houses data frame has a column called "NumOfRooms".  
# We try to determine the mean of this column using the following code:
Houses.loc[:,"NumOfRooms"].mean()


import pandas as pd
import matplotlib.pyplot as plt
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data"
BloodDonation = pd.read_csv(url)
BloodDonation.columns = ["MostRecent", "Donations", "Volume", "First", "Donated"]

import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
Iris = pd.read_csv(url, header=None)
Iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

# import packages
import numpy as np
import pandas as pd

# The url for the data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"
Heart = pd.read_csv(url, header=None)
# Replace the default column names 
Heart.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                 "restecg", "thalach", "exang", "oldpeak", "slope",
                 "ca", "thal", "num"]
Heart = Heart.replace(to_replace="?", value=float("NaN"))