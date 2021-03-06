# Import module from parent
import sys; sys.path.append('..')
from mleviatan import *

#Import Digits images (small)
data = np.genfromtxt('../data/digits-small.csv',delimiter=',')

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
LR = LogisticRegression( Xtrain, Ytrain , Xvalidate , Yvalidate )
# Set Learning parameters
LR.Learner.it = 2000
LR.Learner.alpha = 0.00001

LR.Learner.hasConstant = False # Start with advantage?
LR.Learner.hasMomentum = False
LR.Learner.increase = 0.005
LR.Learner.decrease = 0.60

# Learn
LR.Learner.Learn()
