# -*- coding: utf-8 -*-
"""
Created on Fri Feb 05 18:29:34 2016

@author: Shaking
"""

import numpy as np
x1=np.array([[1,2,3],[4,5,6]])
print x1
print np.shape(x1)
xt=np.transpose(x1)
print xt
print x1[:,0]
one=np.ones(np.shape(x1[:,0]))
print one
added=np.append(one,xt)
print added
print (np.shape(xt)[0]+1,np.shape(xt)[1])
final=np.transpose(added.reshape(np.shape(xt)[0]+1,np.shape(xt)[1]))
print final
print np.exp(1)

A=[]
for i in range(10):
    A.append(i)
print A

print range(-3,4)

print 4.234**(0.5)

print one
print np.shape(one)
print one.reshape(np.shape(x1)[0],1)
print np.shape(one.reshape(np.shape(x1)[0],1))
print x1
print np.hstack((one.reshape(np.shape(x1)[0],1),x1))
print np.hstack((x1,x1))
print np.shape(x1)
for i in range(len(x1[:,0])):
    print i
    print x1[i,:]
for i in range(len(x1[0,:])):
    print i
    print x1[:,i]