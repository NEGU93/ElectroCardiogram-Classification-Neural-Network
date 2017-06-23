import get_data as gd
# import som
import numpy as np
from nimblenet.cost_functions import binary_cross_entropy_cost
from nimblenet.activation_functions import sigmoid_function
from nimblenet.neuralnet import NeuralNet
from nimblenet.data_structures import Instance
from nimblenet.learning_algorithms import backpropagation


def print_results(original, guess):
    correctness = (original == guess)
    print('Aciertos totales: ', sum(correctness))
    print('(', '{0:,.2f}'.format(sum(correctness) * 100 / len(correctness)), ')')

    cases = [(0, 'Normales'), (1, 'Ventriculares'), (2, 'Supraventriculares'), (3, 'Nodales Prematuros'), (4, 'otros')]
    falsos_negativos = 0  # Dije que estaba ok cuando no lo estaba
    falsos_positivos = 0  # Dije que estaba mal cuando estaba bien
    totalN = 0
    totalM = 0
    for c, name in cases:
        count = 0
        total = 0
        for i in range(0, len(original)):
            if c == original[i]:
                total += 1
                if c == 0:                  # Add total of healthy cases
                    totalN += 1
                else:                       # Add total of non healthy cases
                    totalM += 1
                if correctness[i]:          # Good diagnose
                    count += 1
                else:                       # Bad diagnose
                    if c == 0:              # I said he was sick but he wasn't
                        falsos_positivos += 1
                    elif guess[i] == 0:     # I said he was OK but he wasn't
                        falsos_negativos += 1
        print name, ' hallados: ',
        print total
        print 'Correctamente: ',
        print count
        print 'Incorrectamente: ',
        print total - count
        print 'Porcentaje correcto: ',
        print '{0:,.2f}'.format(count * 100 / max(total, 1))
        print('------------------------------')
    print 'Falsos positivos: ', falsos_positivos,
    print '{0:,.2f}'.format(falsos_positivos * 100 / max(totalN, 1))
    print 'Falsos negativos: ', falsos_negativos,
    print '{0:,.2f}'.format(falsos_negativos * 100 / max(totalM, 1))
    # import pdb; pdb.set_trace()

wfdb = gd.WFDB()
wfdb.plot_train_cases()
trainingset, training_classes = wfdb.get_training_set()
# data_set, data_classes = wfdb.get_heartbeats()

data_set = wfdb.get_equally_prob_hb(100)
data_set = np.transpose(data_set)
data_classes = [i for i in range(0, 4) for j in range(0, 100)]
# SOM
# som = som.SOM(training_set, length=training_set.shape[1], epochs=7000, x=8, y=8)
# som.plot_weights_trains(training_set, training_classes)
# som.plot_weigths(data_set, data_classes)
# guess = som.get_results(data_set)
# import pdb; pdb.set_trace()

# create the network
settings = {
    # Required settings
    "n_inputs": 250,                                              # Number of network input signals
    "layers": [(250, sigmoid_function), (1, sigmoid_function)],   # [ (number_of_neurons, activation_function) ]
    # Optional settings
    "initial_bias_value": 0.0,
    "weights_low": -0.1,                                        # Lower bound on the initial weight value
    "weights_high": 0.1,                                        # Upper bound on the initial weight value
}
network = NeuralNet(settings)
# Training the net
dataset = [Instance(data_set[i, :], data_classes[i]) for i in range(0, len(data_set))]
# Instance( [inputs], [outputs] )
# Instance([0, 0], [0]), Instance([1, 0], [1]), Instance([0, 1], [1]), Instance([1, 1], [0])

training_set = [Instance(trainingset[i, :], training_classes[i]) for i in range(0, len(trainingset))]
test_set = data_set
cost_function = binary_cross_entropy_cost

backpropagation(
    # Required parameters
    network,                     # the neural network instance to train
    training_set,                # the training dataset
    test_set,                    # the test dataset
    cost_function,               # the cost function to optimize

    # Optional parameters
    ERROR_LIMIT=1e-3,           # Error tolerance when terminating the learning
    max_iterations=(),          # Regardless of the achieved error, terminate after max_iterations epochs. Default: infinite
    batch_size=0,               # Set the batch size. 0 implies using the entire training_set as a batch, 1 equals no batch learning, and any other number dictate the batch size
    input_layer_dropout=0.0,    # Dropout fraction of the input layer
    hidden_layer_dropout=0.0,   # Dropout fraction of in the hidden layer(s)
    print_rate=1000,            # The epoch interval to print progression statistics
    save_trained_network=False  # Whether to ask the user if they would like to save the network after training
)

# print_results(data_classes, guess)

