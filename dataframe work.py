# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# import numpy and pandas for use

import numpy as np
import pandas as pd
# set seed


np.random.seed(42)

# create table for sales

emp_sales = np.random.randint(100,3000, size = (3,4)).astype("float")
print(emp_sales)

sales_df = pd.DataFrame(np.random.randint(100,3000, size = (3,4)), columns = list ('ABCD')).astype(float)
print(sales_df)

# make up some fake sales people convert to a data frame
info_dict = [{"id":10115, "first_name":"Bob", "last_name": "Edwards", "comm_pct":5.6}, {"id": 10117,
             "first_name":"Helen", "last_name":"Sanchez", "comm_pct":7.8},
{ "id":19928, "first_name":"Mohammed", "last_name":"Mamali", "comm_pct":5.6}  ]
info_df = pd.DataFrame(info_dict)
# oops, forgot an id column in the sales data frame
sales_df['id'] = info_df['id']
print(sales_df)

# or just assign explicitly

sales_df['id']= [10115, 10117, 19928]
print(sales_df)

#let's give the columns new names
sales_df.columns = ['Q1 sales', 'Q2 sales', 'Q3 sales', 'Q4 sales', 'id']
print(sales_df)

#what if we didn't know the order of names, or it might change
new_col_names = {"A":"Q1 Sales", "B":"Q2 Sales", "C":"Q3 Sales", "D":"Q4 Sales"}
sales_df.rename(columns=new_col_names, inplace=True)
print(sales_df)
# now for the urge to merge
combo_df = info_df.merge(sales_df, how = "inner", on ="id")
print(combo_df)
# ste an index on the dataframe
combo_df.index = info_df['id']
combo_df.drop('id', axis = 1, inplace = True)
print(combo_df)

#  look at a particular record
print(combo_df.loc[10115])
print(combo_df.loc[10115, ['first_name', 'last_name']])

# first row is numbered 0
print(combo_df.iloc[1])
print(combo_df.iloc[0])

# add a row
combo_df.loc[19917] = [4.6,"Astrid","Moonbeam", 553, 1912, 198, 2055]
print(combo_df)

#now add a total sales column
combo_df['total sales'] = combo_df[['Q1 sales','Q2 sales','Q3 sales','Q4 sales']] .sum(axis=1)
print(combo_df)
print(combo_df.iloc[0])
# now find the commission
combo_df['commission'] = combo_df['comm_pct'] * combo_df['total sales'] * 0.01
print(combo_df)

R = pd.DataFrame(np.random.randint(1,10, size =(4,2)), columns =list('ab')).astype(float)
print(R)
R['a']= [1,2,3,4]
R['b'] = ['A','B', 'A','B']
S =pd.DataFrame(np.random.randint(1,10, size =(2,1)), columns =list('a')).astype(float)
print(S)
S['a'] = [1,3]
print(S)

T = R.merge(S, how = "full", on = "a")

print(T)