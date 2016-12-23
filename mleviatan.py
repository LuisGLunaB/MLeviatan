import numpy as np
import math

data = np.genfromtxt('casas.csv',delimiter=',')
#data = np.matrix(data)

# Debugging and Optimization
def get_data_base(arr):
    '''
    For a given Numpy array, finds the
    base array that "owns" the actual data.
    '''
    base = arr
    while isinstance(base.base, np.ndarray):
        base = base.base
    return base
def is_same(x, y):
    return get_data_base(x) is get_data_base(y)

# Data Wrangling and Preparation
def split_X_Y( data , n = 1 ):
    '''
    Splits training data into features and labels.
    Inputs:
        data: NP, Unsplitted data.
        [n]: int, Number or labels.
    Ouputs:
        ( X , Y ): NPs, feature and labels respectively.
    '''
    X = data[ : , 0:-n ]
    Y = data[ : , -n: ]
    return ( X , Y )
def split_holdout( data , p = 0.3 , shuffle = False):
    '''
    Separates datasets into training and crossvalidation sets
    using the Hold-out the method.
    Inputs:
        data: NP, Unsplitted data.
        [p]: float, Percentage of data corresponding to the test set.
        [shuffle]: bool, Determines if selection should be random or as given,
                   shuffling copies data to new NP, and thus, lowers performance.
    Outputs:
        Dtrain: NP, Training set.
        Dvalidate: NP, Test set.
        Itrain: [ints], List of elements corresponding to the training set.
        Ivalidate: [ints], List of elements corresponding to the test set.
    '''
    #Generate vector of random integers in the the range of 0 to m.
    (m , n) = data.shape
    v = np.arange(m)
    if shuffle: np.random.shuffle(v)

    #Get Lists of elements.
    split = int(math.floor(m*p))
    Itrain = v[ split:  ]
    Ivalidate = v[ 0:split ]

    #Separate set into training and tesing using both lists.
    if shuffle:
        Dtrain = np.take(data, Itrain, axis=0)
        Dvalidate = np.take(data, Ivalidate, axis=0)
    else:
        Dtrain = data[ split:, : ]
        Dvalidate = data[ 0:split , : ]

    return (Dtrain,Dvalidate,Itrain,Ivalidate)

#np.random.shuffle(data)
( Dtrain , Dvalidate , Itrain , Ivalidate ) = split_holdout(data)
( Xtrain , Ytrain ) = split_X_Y( Dtrain )
( Xvalidate , Yvalidate ) = split_X_Y( Dvalidate )

print Ivalidate
