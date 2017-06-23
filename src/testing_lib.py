from nimblenet.activation_functions import sigmoid_function
from nimblenet.cost_functions import cross_entropy_cost
from nimblenet.learning_algorithms import RMSprop
from nimblenet.data_structures import Instance
from nimblenet.neuralnet import NeuralNet


dataset        = [
    Instance( [0,0], [0] ), Instance( [1,0], [1] ), Instance( [0,1], [1] ), Instance( [1,1], [0] )
]

settings       = {
    "n_inputs" : 2,
    "layers"   : [  (5, sigmoid_function), (1, sigmoid_function) ]
}

network        = NeuralNet( settings )
training_set   = dataset
test_set       = dataset
cost_function  = cross_entropy_cost


RMSprop(
        network,                            # the network to train
        training_set,                      # specify the training set
        test_set,                          # specify the test set
        cost_function,                      # specify the cost function to calculate error

        ERROR_LIMIT             = 1e-2,     # define an acceptable error limit
        #max_iterations         = 100,      # continues until the error limit is reach if this argument is skipped
    )