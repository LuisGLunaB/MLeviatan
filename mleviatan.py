import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
import sys
import inspect

# Debugging and Optimization
def times(X,w,constant = False):
    if not constant:
        R = np.dot(X,w)
    else:
        c = w[0,:]
        R = np.dot(X,w[1:,:]) + c
    return R
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
def printp(i,it,TrainError,ValidateError,running = True):
    text = "Learning ({0} of {1}): T:{2} V:{3}".format( i ,it,TrainError,ValidateError)
    if not running or i == 0:
        print text
    else:
        print "\r" , text ,
def sshape(X):
    (m,n) = X.shape
    print m , n

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
def XV_LearningCurve( X , Y , MLmodel, K):
    ErrorTrain = np.empty([K-1, 1])
    ErrorValidate = np.empty([K-1, 1])
    Models =  ([None] * (K-1) )
    i = 2
    ( _ , MLmodel.Xvalidate , _ , _ ) = split_Kfolds( X , K , 1 )
    ( _ , MLmodel.Yvalidate, _ , _ ) = split_Kfolds( Y , K , 1 )
    ( _ , MLmodel.Xtrain , _ , _ ) = split_Kfolds( X , K , i )
    ( _ , MLmodel.Ytrain , _ , _ ) = split_Kfolds( Y , K , i )
    ( ErrorTrain[i-2][0] , ErrorValidate[i-2][0] ) = MLmodel.EvaluateXV( True )
    Models[1] = MLmodel.GetModel()
    for i in list(range(3,K+1)):
        ( _ , Xtrain , _ , _ ) = split_Kfolds( X , K , i )
        ( _ , Ytrain , _ , _ ) = split_Kfolds( Y , K , i )
        MLmodel.Xtrain = np.concatenate( (MLmodel.Xtrain, Xtrain) , axis=0)
        MLmodel.Ytrain = np.concatenate( (MLmodel.Ytrain, Ytrain) , axis=0)
        ( ErrorTrain[i-2][0] , ErrorValidate[i-2][0] ) = MLmodel.EvaluateXV( True )
        Models[i-2] = MLmodel.GetModel()
    return ( ErrorTrain, ErrorValidate , Models)
def XV_LearningCurveN( X , Y , MLmodel, K, it):
    (m,n) = Y.shape
    ErrorTrain =  np.zeros((K-1, 1))
    ErrorValidate =  np.zeros((K-1, 1))
    for i in list(range(it)):
        v = np.arange(m)
        np.random.shuffle(v)
        Xv = np.take(X, v, axis=0)
        Yv = np.take(Y, v, axis=0)
        (Train, Test, _ ) = XV_LearningCurve( Xv , Yv , MLmodel, K)
        ErrorTrain = ErrorTrain + np.copy(Train)
        ErrorValidate = ErrorValidate + np.copy(Test)
    ErrorTrain = (ErrorTrain / it)
    ErrorValidate = (ErrorValidate / it)
    return ( np.array(ErrorTrain), np.array(ErrorValidate) )

def split_holdout( data , p = 0.33 , shuffle = False):
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
def XV_holdout( X , Y , MLmodel, p = 0.33 ):
    ( MLmodel.Xtrain , MLmodel.Xvalidate , _ , _ ) = split_holdout( X , p )
    ( MLmodel.Ytrain , MLmodel.Yvalidate, _ , _ ) = split_holdout( Y , p )
    ( Train , Validate ) = MLmodel.EvaluateXV( True )
    Models = MLmodel.GetModel()

    return ( Train, Validate, None , None , Models)

def split_Kfolds( data , K , i = 1 ):
    (m , a) = data.shape
    N = int(math.floor(m/K))

    Ivalidate = np.arange( ((i-1)*N) , (i*N) )
    Dvalidate = np.take(data, Ivalidate, axis=0)

    Itrain =  np.append( np.arange(0,(i-1)*N) , np.arange((i*N) , (N*K)) )
    Dtrain = np.take( data , Itrain, axis=0)

    return (Dtrain,Dvalidate,Itrain,Ivalidate)
def XV_Kfolds( X , Y , MLmodel, K):
    ErrorTrain = np.empty([K, 1])
    ErrorValidate = np.empty([K, 1])
    Models =  ([None] * K)
    for i in list(range(1,K+1)):
        ( MLmodel.Xtrain , MLmodel.Xvalidate , _ , _ ) = split_Kfolds( X , K , i )
        ( MLmodel.Ytrain , MLmodel.Yvalidate, _ , _ ) = split_Kfolds( Y , K , i )
        ( ErrorTrain[i-1][0] , ErrorValidate[i-1][0] ) = MLmodel.EvaluateXV( True )
        Models[i-1] = MLmodel.GetModel()
    Train = ErrorTrain.mean()
    Validate = ErrorValidate.mean()
    return ( Train, Validate, ErrorTrain, ErrorValidate , Models)

def split_leave1out( data , x = 0 , m = None ):
    if m is None: (m , a) = data.shape

    Ivalidate = np.array([x])
    Dvalidate = np.take(data, Ivalidate, axis=0)

    Itrain =  np.append( np.arange(0,x) , np.arange((x+1),m) )
    Dtrain = np.take( data , Itrain, axis=0)

    return (Dtrain,Dvalidate,Itrain,Ivalidate)
def XV_leave1out( X , Y , MLmodel ):
    (m , n) = X.shape
    ErrorTrain = np.empty([m, 1])
    ErrorValidate = np.empty([m, 1])
    Models =  ([None] * m)
    for i in list(range(m)):
        ( MLmodel.Xtrain , MLmodel.Xvalidate , _ , _ ) = split_leave1out( X , i , m )
        ( MLmodel.Ytrain , MLmodel.Yvalidate, _ , _ ) = split_leave1out( Y , i , m )
        ( ErrorTrain[i][0] , ErrorValidate[i][0] ) = MLmodel.EvaluateXV( True )
        Models[i] = MLmodel.GetModel()
    Train = ErrorTrain.mean()
    Validate = ErrorValidate.mean()
    return ( Train, Validate, ErrorTrain, ErrorValidate , Models)
def add_constant(X):
    (m,_) = X.shape
    ones =  np.ones((m, 1))
    return np.concatenate( (ones,X) , axis=1)
def rem_constant(X):
    return X[:,1:]

def randomWeights(n,o, base = 1.0):
    return (np.array( np.random.random_sample( (n,o) ))*2-1)*base
def Normalize(X, mean = None, std = None):
    if mean is None: mean = X.mean(axis = 0)
    if std  is None: std = X.std(axis = 0)
    deviation = X - mean
    return ( np.divide(deviation,std) , mean , std )
def NormalizeOne(X, mini = None, maxi = None):
    if maxi is None: maxi = X.max(axis = 0)
    if mini  is None: mini = X.min(axis = 0)
    diff = maxi - mini
    deviation = X - mini
    return ( np.divide(deviation,diff) , mini , maxi )

# Activation Functions
def inverse(f):
    if f == linearActivation: inv = linearActivation
    if f == sigmoidActivation: inv = sigmoidInverse
    return inv
def derivative(f):
    if f == linearActivation: der = linearActivation
    if f == sigmoidActivation: der = sigmoidDerivative
    if f == RectifierActivation: der = RectifierDerivative
    if f == tanhActivation: der = tanhDerivative
    return der
def inverseData(f,X):
    inv = inverse(f)
    if inv != linearActivation:
        X = inv(X)
    return X
def linearActivation(X):
    return X
def RectifierActivation(X):
    R = np.log( np.exp(X)+1 )
    return R
def RectifierDerivative(X):
    return sigmoidActivation(X)
def sigmoidActivation(X):
    np.seterr(over='ignore')
    R = 1 / (1 + np.exp(-X) )
    np.seterr(over='warn')
    return R
def sigmoidDerivative(Z):
    s = sigmoidActivation(Z)
    return (s)*(1-s)
def sigmoidInverse(X):
    R = ( safelog(X) - safelog(1-X) )
    return R
def logit(X):
    R = ( safelog(X) - safelog(1-X) )
    return R

def tanhActivation(X):
    return np.tanh(X)
def tanhDerivative(X):
    return (1-tanhActivation(X))
# Error Functions
def MAD(P,Y):
    return np.mean(np.absolute(P-Y))
def MSE(P,Y):
    return np.mean(np.square(P-Y))
def RMSE(P,Y):
    return ( MSE(P,Y) ) ** 0.5
def LogLoss(P,Y):
    LL = Y * safelog(P)
    return -( LL.mean() )
def Difference(P,Y):
    return (Y-P)
def safelog(X, minval=0.00000001):
    maxval = 1 - minval
    R = np.log( np.clip(X,minval,maxval) )
    return R

class DefaultModeler(object):
    """Empty Modeler"""
    def __init__(self):
        pass
class History(object):
    """Empty Modeler"""
    def __init__(self):
        pass
class ForwardPassPredictor(object):
    """Matrix multiplication predictor"""
    def __init__(self, parent = None ):
        self.parent = parent
        self.activation = linearActivation
    def Predict(self,X = None):
        if X is None: X = self.parent.Xtrain
        w = self.parent.Modeler.w

        Prediction = X.dot(w)

        if self.activation != linearActivation:
            Prediction = self.activation(Prediction)

        return Prediction
class NormalEquationLearner(object):
    """Normal Equation Learner for Linear Regression"""
    def __init__(self, parent = None ):
        self.parent = parent
    def Learn(self):
        #Learning
        Xtranspose = self.parent.Xtrain.transpose()
        #self.w = np.linalg.inv( (Xtranspose.dot(self.parent.Xtrain)) ).dot( Xtranspose.dot(self.parent.Ytrain) )
        #lstsq
        self.w = np.linalg.lstsq( self.parent.Xtrain, self.parent.Ytrain )
        self.w = self.w[0]
        #Take model from the learning
        Modeler = DefaultModeler()
        Modeler.w = self.w
        return Modeler
class LRGradientLearner(object):
    """Gradient Descent for Regression"""
    def __init__(self, parent = None ):
        self.parent = parent
        self.ErrFunction = Difference
        self.Init_size = 0.05
        self.alpha = 0.00000001

        self.ShowProgress = True
        self.RecordWeight = False
        self.RecordGradient = False
        self.RecordAlpha = False

        self.hasConstant = True

        self.it = 500
        if self.it is not None:
            self.Convergence = 0.000000000001

    def Prepare(self):
        # Declare Constants
        self.Y = self.parent.Ytrain
        self.Xtranspose = self.parent.Xtrain.transpose()
        (n,_) = self.Xtranspose.shape
        (_,o) = self.Y.shape

        # Create Modeler
        self.parent.Modeler = DefaultModeler()
        self.parent.Modeler.w = np.array( np.random.random_sample( (n,o) ))*2-1
        self.parent.Modeler.w = self.parent.Modeler.w * self.Init_size

        # Create History variables
        self.History = History()
        self.History.TrainError = []
        self.History.ValidateError = []
        if self.RecordWeight: self.History.w = []
        if self.RecordGradient: self.History.gradient = []
        if self.RecordAlpha: self.History.alpha = []

        # Start with advantage
        self.Advantage()
    def Advantage(self):

        if self.hasConstant:
            f = self.parent.Predictor.activation
            averageY = self.Y.mean(axis = 0)
            self.parent.Modeler.w[0,:] = inverseData( f , averageY )

    def findAlpha(self , trialw, th = 0.5, steps = 3, baseFunc = MAD):
        self.Descent()
        baseError = baseFunc(self.parent.TrainingPrediction,self.parent.Ytrain)

        for i in list(range(250)):
            self.parent.Modeler.w = trialw
            self.Update()
            self.parent.PredictTrain()
            trialError = baseFunc(self.parent.TrainingPrediction,self.parent.Ytrain)
            trialError1 =  (baseError - trialError) * steps

            self.parent.Modeler.w = trialw
            self.Update( self.alpha * steps )
            self.parent.PredictTrain()
            trialError = baseFunc(self.parent.TrainingPrediction,self.parent.Ytrain)
            trialError10 =  (baseError - trialError)
            #print trialError1 , trialError10

            trialRate = ( trialError10 / trialError1 )
            #change = (previoustrialRate-trialRate)

            if (trialRate > th):
                #print i , self.alpha , trialRate
                self.alpha = self.alpha * 1.1
            else:
                if self.ShowProgress:
                    print "Learning Rate:", self.alpha
                break
        self.parent.Modeler.w = trialw

    def Momentum(self, increase = 0.03 , decrease = 0.3, alpha = None, currentGradient = None, previousGradient = None ):
        if previousGradient is None: previousGradient = self.previousGradient
        if currentGradient is None: currentGradient = self.gradient
        if alpha is None: alpha = self.alpha

        product = currentGradient * previousGradient
        selection = product > 0.0
        adition = (selection * alpha) * increase
        substraction = (-selection * alpha) * decrease
        self.alpha = alpha + adition - substraction
        #print self.alpha
        return self.alpha

    def Learn(self):
        # Prepare Variables
        self.Prepare()
        trialw = self.parent.Modeler.w

        self.findAlpha( trialw )
        #Main Iterator
        i = 0
        while True:
            self.previousGradient = self.gradient
            self.Descent()
            self.Momentum()
            self.Update()

            self.RecordProgress(i, running = True )

            # Check for Convergence
            if self.it is not None:
                if i >= self.it : break
            elif i > 1 :
                Improvement = -( (self.History.TrainError[i] / self.History.TrainError[i-1]) - 1 )
                if self.Convergence > Improvement : break
            i = i + 1
        print ""
        return self.parent.Modeler

    def Descent(self):
        self.parent.EvaluateValidate()
        self.parent.EvaluateTrain()
        self.gradient = self.Gradient()
        return self.gradient
    def Gradient(self):
        self.A = self.ErrFunction(self.parent.TrainingPrediction,self.Y)
        self.gradient = self.Xtranspose.dot(self.A)
        return self.gradient
    def Update(self, alpha = None , gradient = None ):
        if gradient is None: gradient = self.gradient
        if alpha is None: alpha = self.alpha
        Update = (gradient * alpha)
        self.parent.Modeler.w = self.parent.Modeler.w + Update
        return Update
    def RecordProgress(self,i, running = True):
        self.History.TrainError.append(self.parent.TrainError)
        self.History.ValidateError.append(self.parent.ValidateError)

        if self.RecordWeight: self.History.w.append( self.parent.Modeler.w )
        if self.RecordGradient: self.History.gradient.append( self.gradient )
        if self.RecordAlpha: self.History.alpha.append( self.alpha )

        if self.ShowProgress:
            printp( i, self.it, self.History.TrainError[i] , self.History.ValidateError[i] , running )
class NNGradientLearner(LRGradientLearner):
    """Gradient Descent for Neural Networks"""
    def __init__(self,parent = None):
        LRGradientLearner.__init__(self, parent )
        self.alpha = 0.00000001
        self.Init_size = 0.1
        self.gradient = {}
        self.d = {}
        self.hasConstant = False
    def Prepare(self):
        L = self.parent.Layers
        self.parent.RandomRestart(self.Init_size)
        # Create History variables
        self.History = History()
        self.History.TrainError = []
        self.History.ValidateError = []
        if self.RecordWeight: self.History.w = []
        if self.RecordGradient: self.History.gradient = []
        if self.RecordAlpha: self.History.alpha = []

        # Start with advantage
        self.Advantage()
    def Advantage(self):
        L = self.parent.Layers
        if self.hasConstant:
            f = self.parent.Modeler.activation[L-2]
            averageY = self.parent.Ytrain.mean(axis = 0)
            self.parent.Modeler.w[L-2][0,:] = inverseData( f , averageY )

    def Gradient(self):
        L = self.parent.Layers
        self.A = self.ErrFunction(self.parent.TrainingPrediction,self.parent.Ytrain)
        BackProp = self.A
        for l in list(range(L-1,0,-1)): #[2,1]
            derive = derivative(self.parent.Modeler.activation[l-1])
            self.d[l] = derive( self.parent.Hidden[l] ) * BackProp

            self.gradient[l-1] = np.transpose(self.parent.Hidden[l-1]).dot(self.d[l])
            if l > 1: BackProp = self.d[l].dot( np.transpose(self.parent.Modeler.w[l-1]) )
        #self.gradient = self.Xtranspose.dot(self.A)
        return self.gradient
    def Update(self, alpha = None , gradient = None ):
        #print self.parent.Modeler.w[1]
        L = self.parent.Layers
        Update = {}
        if gradient is None: gradient = self.gradient
        if alpha is None: alpha = self.alpha
        for l in list(range(L-1)):
            Update[l] = (gradient[l] * alpha)
            self.parent.Modeler.w[l] = self.parent.Modeler.w[l] + Update[l]
        return Update
    def Learn(self):
        self.Prepare()

        i = 0
        while True:
            self.previousGradient = self.gradient
            self.Descent()
            #self.Momentum()
            self.Update()

            self.RecordProgress(i, running = True )
            # Check for Convergence
            if self.it is not None:
                if i >= self.it : break
            elif i > 1 :
                Improvement = -( (self.History.TrainError[i] / self.History.TrainError[i-1]) - 1 )
                if self.Convergence > Improvement : break
            i = i + 1

            #print self.parent.PreHidden[1].std()

        print ""
        return self.parent.Modeler

# Machine Learning Algorithms Classes
# np.concatenate( (Train,Test) , axis=1)
class DefaultML(object):
    """Default Machine Learning Model"""
    def __init__(self, Xtrain = None, Ytrain = None, Xvalidate = None, Yvalidate = None ):
        self.Name = "Default Machine Learning Model"
        self.Alias = ""
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xvalidate = Xvalidate
        self.Yvalidate = Yvalidate

        self.Learner = None
        self.Modeler = None
        self.Predictor = None

        self.Prediction = None
        self.TrainingPrediction = None
        self.ValidationPrediction = None

        self.ErrorFunction = None
        self.Error = None
        self.TrainError = None
        self.ValidateError = None

    def describe(self):
        print "\n" + self.Name +"("+self.Alias+"):","\n"

        print "Modeler:" , type(self.Modeler).__name__
        print "Predictor:" , type(self.Predictor).__name__,"\n"

        print self.ErrorFunction.__name__ ,"Error:" , self.Error
        print self.ErrorFunction.__name__ ,"Train error:" , self.TrainError
        print self.ErrorFunction.__name__ ,"Validation error:" , self.ValidateError,"\n"

        print "Model:\n" , self.GetModel(),"\n"

        print "Learning setup:\n" , self.GetLearning()

    def SetLearner(self,LearnerObject):
        if inspect.isclass(LearnerObject):
            self.Learner = LearnerObject(self)
        else:
            self.Learner = LearnerObject
            self.Learner.parent = self
    def SetModeler(self,ModelerObject):
        self.Modeler = ModelerObject
    def SetPredictor(self,PredictorObject):
        self.Predictor = PredictorObject(self)

    def GetModel(self):
        if self.Modeler is None: self.Learn()
        return (self.Modeler.__dict__)
    def GetLearning(self):
        string = "Algorithm: " + type(self.Learner).__name__ + "\n"
        setup = self.Learner.__dict__
        for (key,value) in setup.iteritems():
            if not key == "parent" and not key == "w":
                string = "" + key + ": ",value
        return string

    def Learn(self):
        self.Modeler = self.Learner.Learn()

    def Predict(self,learn = False):
        if self.Modeler is None or learn: self.Learn()
        self.Prediction = self.Predictor.Predict(self.Xtrain)
        return self.Prediction
    def PredictTrain(self, learn = False):
        if learn: self.Learn()
        self.TrainingPrediction = self.Predictor.Predict(self.Xtrain)
        return self.TrainingPrediction
    def PredictValidate(self):
        self.ValidationPrediction = self.Predictor.Predict(self.Xvalidate)
        return self.ValidationPrediction
    def PredictXV(self,learn = False):
        self.PredictTrain(learn)
        self.PredictValidate()
        return ( self.TrainingPrediction , self.ValidationPrediction )
    def Evaluate(self, learn = False , predict = True):
        if self.Prediction is None or learn or predict: self.Predict(learn)
        self.Error = self.ErrorFunction(self.Prediction,self.Ytrain)
        return self.Error
    def EvaluateTrain(self, learn = False, predict = True ):
        if predict: self.PredictTrain(learn)
        self.TrainError = self.ErrorFunction(self.TrainingPrediction,self.Ytrain)
        return self.TrainError
    def EvaluateValidate(self, predict = True ):
        if predict: self.PredictValidate()
        self.ValidateError = self.ErrorFunction(self.ValidationPrediction,self.Yvalidate)
        return self.ValidateError
    def EvaluateXV(self, learn = False, predict = True ):
        self.EvaluateTrain(learn, predict)
        self.EvaluateValidate(predict)
        return (self.TrainError,self.ValidateError)
class LinearRegression(DefaultML):
    def __init__(self, Xtrain = None, Ytrain = None, Xvalidate = None, Yvalidate = None ):
        """Linear Regression Model"""
        DefaultML.__init__(self, Xtrain, Ytrain, Xvalidate, Yvalidate)
        self.Name = "Linear Regression Model"
        self.Alias = ""
        self.Modeler = None
        self.Learner = NormalEquationLearner(self)
        self.Learner.ShowProgress = True
        self.Predictor = ForwardPassPredictor(self)
        self.ErrorFunction = RMSE
class LogisticRegression(LinearRegression):
    def __init__(self, Xtrain = None, Ytrain = None, Xvalidate = None, Yvalidate = None ):
        """Logistic Regression Model"""
        LinearRegression.__init__(self, Xtrain, Ytrain, Xvalidate, Yvalidate)
        self.Name = "Logistic Regression Model"
        self.Learner = LRGradientLearner(self)
        self.Predictor.activation = sigmoidActivation
        self.ErrorFunction = LogLoss

class FFPredictor(object):
    """Neural Network FeedForward"""
    def __init__(self, parent = None ):
        self.parent = parent
        self.L = self.parent.Layers
        for l in list(range(self.L-1)):
            self.parent.Modeler.activation[l] = sigmoidActivation
        self.parent.Modeler.activation[0] = tanhActivation
    def Predict(self,X = None):
        if X is None: X = self.parent.Xtrain
        self.parent.Hidden[0] = X
        self.parent.PreHidden[0] = X
        w = self.parent.Modeler.w
        for l in list(range(self.L-1)):
            data = self.parent.Hidden[l]
            weight = w[l]
            self.parent.PreHidden[l+1] = data.dot(weight)
            if self.parent.Modeler.activation[l] == linearActivation:
                self.parent.Hidden[l+1] = self.parent.PreHidden[l+1]
            else:
                self.parent.Hidden[l+1] = self.parent.Modeler.activation[l]( self.parent.PreHidden[l+1] )
        Prediction = self.parent.Hidden[self.L-1]
        return Prediction

class NeuralNetwork(DefaultML):
    def __init__(self, Xtrain = None, Ytrain = None, Xvalidate = None, Yvalidate = None ):
        """Feed Forward Neural Network Model"""
        DefaultML.__init__(self, Xtrain, Ytrain, Xvalidate, Yvalidate)
        self.Name = "Feed Forward Neural Network Model"
        self.Alias = ""

        self.Modeler = DefaultModeler()
        self.Modeler.w = {}
        self.Modeler.Units = {}
        self.Modeler.activation = {}

        self.ErrorFunction = LogLoss
        self.Layers = 3
        self.Predictor = FFPredictor(self)
        self.Units = {}

        self.PreHidden = {}
        self.PreHidden[0] = self.Xtrain
        self.Hidden = {}
        self.Hidden[0] = self.Xtrain

        self.Learner = NNGradientLearner(self)
        self.Learner.ShowProgress = True

    def RandomRestart(self, base = 1.0 ):
        #( _ ,self.Units[0]) = self.Xtrain.shape
        #( _ ,self.Units[self.Layers+1]) = self.Ytrain.shape
        self.Modeler.w = {}
        for l in list(range(self.Layers-1)):
            self.Modeler.w[l] = randomWeights(self.Modeler.Units[l],self.Modeler.Units[l+1], base )

#Import Digits images (small)
data = np.genfromtxt('./data/digits-small.csv',delimiter=',')

# Separate data in 2 blocks
( Dtrain , Dvalidate , Itrain , Ivalidate ) = split_holdout(data)
# Separte X from Y
( Xtrain , Ytrain ) = split_X_Y( Dtrain , n=10 )
( Xvalidate , Yvalidate ) = split_X_Y( Dvalidate, n=10 )
# Normalize Images and Add Constant
(Xtrain, mean , std ) = NormalizeOne(Xtrain, 0, 255)
(Xvalidate, _ , _ ) = NormalizeOne(Xvalidate, 0, 255)
Xtrain = add_constant(Xtrain)
Xvalidate = add_constant(Xvalidate)

# Create ML Object
NN = NeuralNetwork( Xtrain, Ytrain , Xvalidate , Yvalidate )
NN.Modeler.Units = [785,30,10]
NN.Learner.alpha = 0.0001
NN.Learner.it = 4000
NN.Learner.Learn()


'''
#(Train, Test ) = XV_LearningCurveN( X, Y, LR, 54, 3000)
plt.title('Learning Curve')
plt.ylabel('Error')
plt.xlabel('Folds')
plt.plot(Train[6:], label="Training Error")
plt.plot(Test[6:], label="Validation Error")
plt.legend()
plt.show()
'''
