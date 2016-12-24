import numpy as np
import weakref
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

# Common Error functions
def MAD(P,Y):
    return np.mean(np.absolute(P-Y))
def MSE(P,Y):
    return np.mean(np.square(P-Y))
def RMSE(P,Y):
    return ( MSE(P,Y) ) ** 0.5

class DefaultModeler(object):
    """Empty Modeler"""
    def __init__(self):
        pass

class ForwardPassPredictor(object):
    """Matrix multiplication predictor"""
    def __init__(self):
        self.parent = None
    def Predict(self,X = None):
        if X is None: X = self.parent.Xtrain
        Prediction = X.dot(self.parent.Modeler.w)
        return Prediction

class NormalEquationLearner(object):
    """Normal Equation Learner for Linear Regression"""
    def __init__(self):
        self.parent = None
    def Learn(self):
        #Learning
        self.Xtranspose = self.parent.Xtrain.transpose()
        self.w = np.linalg.inv( (self.Xtranspose.dot(self.parent.Xtrain)) ).dot( self.Xtranspose.dot(self.parent.Ytrain) )
        #Take model from the learning
        Modeler = DefaultModeler()
        Modeler.w = self.w
        return Modeler

# Linear Regression Object
class LinearRegression(object):
    """Linear Regression Model"""
    def __init__(self, Xtrain = None, Ytrain = None, Xvalidate = None, Yvalidate = None , ErrorFunction = RMSE ):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xvalidate = Ytrain
        self.Yvalidate = Yvalidate

        self.Learner = None
        self.Modeler = None
        self.Predictor = None

        self.Prediction = None
        self.TrainingPrediction = None
        self.ValidationPrediction = None

        self.ErrorFunction = ErrorFunction
        self.Error = None
        self.TrainError = None
        self.ValidateError = None

    def SetLearner(self,LearnerObject):
        self.Learner = LearnerObject
        self.Learner.parent = self
    def SetModeler(self,ModelerObject):
        self.Modeler = ModelerObject
    def SetPredictor(self,PredictorObject):
        self.Predictor = PredictorObject
        self.Predictor.parent = self

    def Collect(self):
        if self.Prediction is None: self.Predict()
        return self.Prediction

    def Learn(self):
        self.Modeler = self.Learner.Learn()

    def Predict(self):
        if self.Modeler is None: self.Learn()
        self.Prediction = self.Predictor.Predict()
        return self.Prediction
    def PredictTrain(self):
        self.TrainingPrediction = self.Predictor.Predict(self.Xtrain)
        return self.TrainingPrediction
    def PredictValidate(self):
        self.ValidationPrediction = self.Predictor.Predict(self.Xvalidate)
        return self.ValidationPrediction

    def Evaluate(self):
        if self.Prediction is None: self.Predict()
        self.Error = self.ErrorFunction(self.Prediction,self.Ytrain)
        return self.Error
    def EvaluateTrain(self):
        self.TrainError = self.ErrorFunction(self.TrainingPrediction,self.Ytrain)
        return self.TrainError
    def EvaluateValidate(self):
        self.ValidateError = self.ErrorFunction(self.ValidationPrediction,self.Yvalidate)
        return self.ValidateError

#np.random.shuffle(data)
( Dtrain , Dvalidate , Itrain , Ivalidate ) = split_holdout(data)
( Xtrain , Ytrain ) = split_X_Y( Dtrain )
( Xvalidate , Yvalidate ) = split_X_Y( Dvalidate )

( X , Y ) = split_X_Y( data )

LR = LinearRegression( X, Y )
FP = ForwardPassPredictor()
LR.SetPredictor(FP)

NE = NormalEquationLearner()
LR.SetLearner(NE)

LR.Learn()
Modelo = LR.Modeler
LR.SetModeler(Modelo)

print LR.Predict()
