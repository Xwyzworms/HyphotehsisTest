# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:50:30 2020

@author: XywzWorm
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse



MAX_DATASET = 50
NOISE_VARIANCE = 0.5 ##IRREDUCABLE ERRORS --> Can't Fix This shit
MAX_DEGREE = 12 ##Max Degree For Poly
N = 25 ## total Datapoints 
N_TRAIN= int (0.9 * N)

np.random.seed(2)

## Make a DataSet

def create_Poly(x,Degree):
    N = len(x)
    X = np.empty( ( N,Degree+1 ) ) 
    for degree in range(Degree + 1):
        X[: , degree] = x ** degree
        if( degree > 1 ):
            ## Do Normalization ,cuz the Value is high,lol
            X[:,degree] =  ( X[:,degree] - X[:,degree].mean() ) / X[:,degree].std()
    return X        

def f(x):
    return np.sin(x)







if __name__ == "__main__":
    
    ## Creating Axis
    x_axis = np.linspace(-np.pi , np.pi , 100)
    y_axis=f(x_axis)
    
    #  Creating dataset
    X = np.linspace(-np.pi , np.pi , N)
    np.random.shuffle(X)
    ## Make Sinus of (X)
    f_x= f(X)
    
    XPolynom=create_Poly(X,MAX_DEGREE)
    
    ## create Array For Storing the result
    Train_scores =  np.zeros( ( MAX_DATASET , MAX_DEGREE ) )
    Test_scores  =  np.zeros(  ( MAX_DATASET , MAX_DEGREE ) )
    Train_Prediction_scores = np.zeros( ( N_TRAIN , MAX_DATASET , MAX_DEGREE ) ) ## 3D Array Since We want to get bias,variance And Irreducible Erros
    Prediction_Curves = np.zeros( (100, MAX_DATASET , MAX_DEGREE ) ) ## Only for Plots 
    print(Prediction_Curves.shape)
    model=LinearRegression()
    for numdataset in range(MAX_DATASET):
        Y = f_x + np.random.rand(N) * NOISE_VARIANCE
        XTrain = XPolynom[:N_TRAIN]
        Ytrain = Y[:N_TRAIN]
    
        XTest = XPolynom[N_TRAIN:]
        YTest = Y[N_TRAIN:] ## 10 %  of Y values        
        #print("This Y Test" ,YTest)
        for degree in range(MAX_DEGREE):
            model.fit(XTrain[: , :degree +2] , Ytrain)
            predictions=model.predict(XPolynom[: ,:degree +2 ])
            
            x_axis_poly= create_Poly(x_axis , degree + 1) ## Shape ( 100,degree + 1) 
            prediction_axis=model.predict(x_axis_poly) ## Need To be Same Shape ,remember that 
            plt.plot(x_axis,prediction_axis)
        
            Prediction_Curves[:,numdataset,degree] = prediction_axis ### Z Axis ,Columns,Rows
            #print(degree)
            """
            Lol I M tired 
            """
    
    
    
    
    X=np.linspace(-np.pi , np.pi , N )
    lol = create_Poly(X,MAX_DEGREE)
    #print(lol.shape)

