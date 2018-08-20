#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 13:06:49 2018

@author: mutecypher
"""

import numpy as np
x = np.array([5, 7, 9, 53, 4, 11, 4, 8, 7, 12])
Hi = np.mean(x) + 2*np.std(x)
Lo = np.mean(x) - 2*np.std(x)
Good = (x < Hi) & (x > Lo)
y = x[~Good]
y = x[(x >= Hi) | (x <= Lo)]
y = x[(x >= Hi) & (x <= Lo)]
y = x[(x <= Hi) | (x >= Lo)]
y = x[(x <= Hi) & (x >= Lo)]
x[Good] = np.mean(x[Good])
x[~Good] = np.mean(x[~Good])
x[~Good] = np.mean(x[Good])
x[Good] = np.mean(x[~Good])
x = x[Good]
x = x[(x < np.mean(x) + 2*np.std(x)) & (x > np.mean(x) - 2*np.std(x))]
x = np.array([-2, 1, "", 1, 20, 1, 5, -2, "X", -1, 4, 3])
FlagGood = (x != "") & (x != "X")
np.mean(x[FlagGood].astype(int))
np.mean(x[FlagGood].astype(str))
np.mean(x[FlagGood].astype(float))

x = np.array([5, 7, 9, 53, 4, 11, 4, 8, 7, 12, -29])



Hi = np.mean(x) + 2*np.std(x)
Lo = np.mean(x) - 2*np.std(x)
y = x[(x < Hi) & (x > Lo)]
y = x[(x > Hi) &(x < Lo)]
y = x[(x > Hi) | (x < Lo)]


c = "7"
2 + int(c)
a = 5.1
b = np.array([3]) + a
type(b)
int("2" + c)
c = 7
int("2" + c)
c = "7"
int(2 + c)
a = 5
b = a + 3.1
x = np.array([5, -7, 1.1, 1, 99])
x.dtype.name

x = np.array([5, -7, 1, 1, 99])
x.dtype.name
type(b[0])