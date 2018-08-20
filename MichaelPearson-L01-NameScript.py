#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 13:04:51 2018

@author: mutecypher
"""
#Create a function that prints my name - Michael Pearson

def my_name():
    print("Michael Pearson")
    
#This will call the function that prints my name
my_name()

#Now I will import the necessary package for the date and time

import datetime as dt
# To get a better looking format, I found instructions on Stack Overflow
# at  https://stackoverflow.com/questions/7588511/format-a-datetime-into-a-string-with-milliseconds/39890902#39890902
# With some additional guidance from the page https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior

now_is = dt.datetime.now().strftime('%d %B, %Y %I:%M:%S %p')
print(now_is)

#I printed out the date in day/month/year format, with time in 12hour clock form

