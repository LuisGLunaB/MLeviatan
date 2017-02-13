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
NN = NeuralNetwork( Xtrain, Ytrain , Xvalidate , Yvalidate )

NN.ErrorFunction = LogLoss
NN.Learner.activation = [ sigmoidActivation, rectifierActivation, sigmoidActivation]
NN.Learner.it = 10000
NN.Learner.alpha = 0.0001
NN.Learner.Units = [785,50,20,10]

NN.Learner.hasMomentum = True
NN.Learner.increase = 0.0005
NN.Learner.decrease = 0.005

NN.RandomRestart()
NN.Learner.Learn()
