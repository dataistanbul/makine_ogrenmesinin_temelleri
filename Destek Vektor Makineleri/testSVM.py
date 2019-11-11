#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:02:35 2019

@author: zeyneddinoz
"""

import SimSMO


X, Y = SimSMO.loadDataSet('myData.txt')

b, alphas = SimSMO.simplifiedSMO(X, Y, 0.6, 0.001, 40)

w = SimSMO.computeW(alphas, X, Y)


# test with the data point (3, 4)
SimSMO.plotLinearClassifier([3, 4], w, alphas, b, X, Y)

