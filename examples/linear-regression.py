# Import module from parent
import sys; sys.path.append('..')
from mleviatan import *

#Import Digits images (small)
data = np.genfromtxt('../data/housing.csv',delimiter=',')

# Separate data in 2 blocks
( Dtrain , Dvalidate , Itrain , Ivalidate ) = split_holdout(data)
# Separte X from Y
( Xtrain , Ytrain ) = split_X_Y( Dtrain , n=1 )
( Xvalidate , Yvalidate ) = split_X_Y( Dvalidate, n=1 )
# Normalize data and Add Constant
(Xtrain, mean , std ) = Normalize(Xtrain)
(Xvalidate, _ , _ ) = Normalize(Xvalidate, mean, std)
Xtrain = add_constant(Xtrain)
Xvalidate = add_constant(Xvalidate)

# Create ML Object
LR = LinearRegression( Xtrain, Ytrain , Xvalidate , Yvalidate )

# Learn (Normal Equation)
LR.Learn()
LR.EvaluateXV()
print LR.Modeler.w
print LR.TrainError, LR.ValidateError
