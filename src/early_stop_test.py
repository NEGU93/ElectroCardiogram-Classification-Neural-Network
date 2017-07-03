import get_data as gd
import numpy as np
from nimblenet.activation_functions import sigmoid_function
from nimblenet.neuralnet import NeuralNet
from nimblenet.data_structures import Instance
from nimblenet.learning_algorithms import backpropagation
from nimblenet.cost_functions import sum_squared_error

# Parameters
hidden_layer_dropout = 0.8
input_layer_dropout = 0.8
early_stop = 32
num_training_cases = 10
data_reduction = 32
hidden_nodes = 24
num_data_set = None


while early_stop <= 38:
    for _ in xrange(10):

        wfdb = gd.WFDB(num_training_cases, data_reduction)
        wfdb.plot_train_cases()
        trainingset, training_classes = wfdb.get_training_set()
        testset, test_classes = wfdb.get_organized(100)
        testset = np.transpose(testset)

        # create the network
        settings = {
            # Required settings
            "n_inputs": data_reduction,                                              # Number of network input signals
            "layers": [(hidden_nodes, sigmoid_function),
                       (4, sigmoid_function)],   # [ (number_of_neurons, activation_function) ]
            # Optional settings
            "initial_bias_value": 0.0,
            "weights_low": -0.1,                                        # Lower bound on the initial weight value
            "weights_high": 0.1,                                        # Upper bound on the initial weight value
        }
        network = NeuralNet(settings)
        expected_output = [
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
        ]
        # Training the net
        test_set = [Instance(testset[i, :], expected_output[test_classes[i]]) for i in range(0, len(testset))]
        training_set = [Instance(trainingset[i, :], expected_output[training_classes[i]]) for i in range(0, len(trainingset))]
        cost_function = sum_squared_error
        print 'Starting to train...'
        backpropagation(
            # Required parameters
            network,                     # the neural network instance to train
            training_set,                # the training dataset
            test_set,                    # the test dataset
            cost_function,               # the cost function to optimize
            # Optional parameters
            ERROR_LIMIT=1e-3,                           # Error tolerance when terminating the learning
            max_iterations=20000,                       # Regardless of the achieved error, terminate after max_iterations epochs. Default: infinite
            batch_size=0,                               # Set the batch size. 0 implies using the entire training_set as a batch, 1 equals no batch learning, and any other number dictate the batch size
            input_layer_dropout=input_layer_dropout,    # Dropout fraction of the input layer
            hidden_layer_dropout=hidden_layer_dropout,  # Dropout fraction of in the hidden layer(s)
            print_rate=1000,                            # The epoch interval to print progression statistics
            save_trained_network=False,                 # Whether to ask the user if they would like to save the network after training
            early_stop=early_stop
        )
    early_stop += 2
