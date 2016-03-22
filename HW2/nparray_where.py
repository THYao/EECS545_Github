# -*- coding: utf-8 -*-
"""
Created on Sun Feb 07 21:39:57 2016

@author: Shaking
"""

import numpy as np
x = np.arange(15).reshape(5,3)
print x
print x>=5
x_where=np.where(x>=5)
print x_where
x_index=x[x>=5]
print x_index

a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print a
# select rows where first column is greater than 3
print a[a[:,0] > 3]
print a[a[:,0] > 3][:,np.array([True, True, False, True])] # select columns
