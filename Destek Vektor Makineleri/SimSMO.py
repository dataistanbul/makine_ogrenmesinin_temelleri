#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:02:35 2019

@authors: zeyneddinoz, ebrutoka
"""

from numpy import random, mat, zeros, multiply, shape
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# #############################################################################

def loadDataSet(fileName):
    
    dataX = []
    labelY = []
    fr = open(fileName)
    
    for r in fr.readlines():
        record = r.strip().split()
        dataX.append([float(record[0]), float(record[1])])
        labelY.append(float(record[2]))
    
    return dataX, labelY

# #############################################################################

def selectJrandomly(i, m):
    
    j = i
    
    while j == i:
        j = int(random.uniform(0, m))
      
    return j


def clipAlphasJ(aj, H, L):
    
    if aj > H:
        aj = H

    # elif
    if L > aj:
        aj = L


    return aj
  
    
def simplifiedSMO(dataX, classY, C, tol, max_passes):

    
    X = mat(dataX)
    Y = mat(classY).T
    m, n = shape(X)
    
    # Initialize b: threshold for solution:
    b = 0
    
    # Initialize alphas: lagrange multipliers for solution:
    alphas = mat(zeros((m, 1)))
    passes = 0
    
    while passes < max_passes:
        
        num_changed_alphas = 0
        
        for i in range(m):
            
            # Calculate Ei = f(xi) - yi
            fXi = float(multiply(alphas,Y).T * (X * X[i,:].T)) + b
            Ei = fXi - float(Y[i])
            
            if (Y[i]*Ei < -tol and alphas[i] < C) or (Y[i]*Ei > tol and alphas[i] > 0):
                
                # select j # i randomly
                j = selectJrandomly(i, m)
                
                # Calculate Ej = f(xj) - yj
                fXj = float(multiply(alphas,Y).T*(X*X[j,:].T)) + b
                Ej = fXj - float(Y[j])
                
                # save old alphas's
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                
                # compute L and H
                if Y[i] != Y[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                
                # if L = H the continue to next i
                if L == H:
                    continue
                
                # compute eta
                eta = 2.0 * X[i,:] * X[j,:].T - X[i,:] * X[i,:].T - X[j,:] * X[j,:].T
                # eta: (Xi-Xj)**2, if eta < 0
                
                # if eta >= 0 then continue to next i
                if eta >= 0:
                    continue
                
                # compute new value for alphas j
                alphas[j] -= Y[j]*(Ei - Ej) / eta
                
                # clip new value for alphas j
                alphas[j] = clipAlphasJ(alphas[j], H, L)
                
                # if |alphasj - alphasold| < 0.00001 then continue to next i
                if abs(alphas[j] - alphaJold) < 0.00001:
                    continue
                
                # determine value for alphas i:
                alphas[i] += Y[j] * Y[i] * (alphaJold - alphas[j])
                
                # compute b1 and b2:
                b1 = b - Ei- Y[i] * (alphas[i] - alphaIold) * (X[i,:] * X[i,:].T) - Y[j] * (alphas[j]-alphaJold) * (X[i,:] * X[j,:].T)
                b2 = b - Ej- Y[i] * (alphas[i] - alphaIold) * (X[i,:] * X[j,:].T) - Y[j] * (alphas[j] - alphaJold) * (X[j,:] * X[j,:].T)
                
                
                # compute b:
                if 0 < alphas[i] and C > alphas[i]:
                    b = b1
                    
                elif 0 < alphas[j] and C > alphas[j]:
                    b = b2
                
                else:
                    b = (b1 + b2) / 2.0                      
                
                num_changed_alphas += 1
                
                
            if num_changed_alphas == 0: 
                passes += 1
                
            else: 
                passes = 0
                

    return b, alphas
   
# #############################################################################
    
def wHesaplama(alfalar, dataX, classY):
    """
    Ağırlıkların hesaplandığı fonksiyondur.
    """
    
    X = mat(dataX)
    Y = mat(classY).T
    
    m, n = shape(X)
    
    w = zeros((n, 1))
    
    for i in range(m):
        w += multiply(alfalar[i]*Y[i], X[i,:].T)
      
        
    return w

# #############################################################################

def sinifTahmini(giris_verisi, w, b):
    """
    w ve b'yi kullanarak giriş verisinin hangi sınıfa ait olduğunu tahminleyen fonksiyon.
    """
    
    p = mat(giris_verisi)
    f = p*w + b
    
    if f > 0:
        print(giris_verisi, "sınıf 1'e ait.")
    
    else:
        print(giris_verisi, "sınıf -1'e ait.")



def plotLinearClassifier(point, w, alphas, b, dataX, labelY):
    
    
    shape(alphas[alphas>0])
      
    Y = np.array(labelY)
    X = np.array(dataX)
    
    svmMat = []
    alphaMat = []
    
    for i in range(10):
        
        alphaMat.append(alphas[i])
        
        if alphas[i] > 0.0:
            svmMat.append(X[i])
                                 
    svmPoints = np.array(svmMat)
    alphasArr = np.array(alphaMat)

    numofSVMs = shape(svmPoints)[0]
    print("Number of SVM points: %d" % numofSVMs)
 
    xSVM = []; ySVM = []
    for i in range(numofSVMs):
        xSVM.append(svmPoints[i, 0])
        ySVM.append(svmPoints[i, 1])    
     
    n = shape(X)[0]    
    
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
      
    for i in range(n):
        if int(labelY[i])== 1:
            xcord1.append(X[i,0])
            ycord1.append(X[i,1])                  
        else:
            xcord2.append(X[i,0])
            ycord2.append(X[i,1])                  

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
       
    for j in range(0, len(xcord1)):
        
        for l in range(numofSVMs):
            
            if xcord1[j]== xSVM[l] and ycord1[j]== ySVM[l]:
                ax.annotate("SVM", 
                            (xcord1[j], ycord1[j]), 
                            (xcord1[j] + 1, ycord1[j] + 2), 
                            arrowprops=dict(facecolor='black', shrink=0.005))
      
        
    ax.scatter(xcord2, ycord2, s=30, c='green')
    
    for k in range(0, len(xcord2)):
        
        for l in range(numofSVMs):
            
            if xcord2[k] == xSVM[l] and ycord2[k] == ySVM[l]:
                ax.annotate("SVM", 
                            (xcord2[k], ycord2[k]), 
                            (xcord2[k] - 1, ycord2[k] + 1), 
                            arrowprops=dict(facecolor='black', shrink=0.005))
      
       
    red_patch = mpatches.Patch(color='red', label='Class 1')
    green_patch = mpatches.Patch(color='green', label='Class -1')
    plt.legend(handles=[red_patch,green_patch])
      
    x = []
    y = []


    for xfit in np.linspace(-3.0, 3.0):
        x.append(xfit)
        y.append(float((-w[0]/w[1])*xfit - b[0,0])/w[1])
        #y.append(float((-w[0]/w[1])*xfit - b)/w[1])
             
    ax.plot(x, y)
      
    predictedClass(point, w, b)
    p = mat(point)
    ax.scatter(p[0,0], p[0,1], s=30, c='black', marker='s')
    circle1 = plt.Circle((p[0,0], p[0,1]), 0.6, color='b', fill=False)
    plt.gcf().gca().add_artist(circle1)
      
    plt.show()
    
# #############################################################################
    

        
        
        
        
        
        
        
        