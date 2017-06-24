import get_data as gd
import som
import numpy as np
from nimblenet.cost_functions import binary_cross_entropy_cost
from nimblenet.activation_functions import sigmoid_function
from nimblenet.neuralnet import NeuralNet
from nimblenet.data_structures import Instance
from nimblenet.learning_algorithms import backpropagation
from nimblenet.cost_functions import sum_squared_error


def print_results(original, guess):
    correctness = np.equal(original, guess)
    cases = [(0, 'Normales'), (1, 'Ventriculares'), (2, 'Supraventriculares'), (3, 'Nodales Prematuros')] # , (4, 'otros')]
    # ponderado = 0
    for c, name in cases:
        count = 0
        total = 0
        for i in range(0, len(guess)):
            if c == guess[i]:
                total += 1
                if correctness[i]:          # Good diagnose
                    count += 1
        totales_reales = sum([1 for o in original if o == c])
        # ponderado += count / max(totales_reales, 1)
        print name, ' totales ',
        print totales_reales
        print name, ' hallados: ',
        print total
        print 'Aciertos: ',
        print count
        print 'Errores: ',
        print total - count
        print 'Porcentaje correcto: ',
        print '{0:,.2f}'.format(count * 100 / max(total, 1)), '%'
        print 'Porcentaje hallados: ',
        print '{0:,.2f}'.format(100 - ((totales_reales - count) * 100 / max(totales_reales, 1))), '%'
        print('------------------------------')
    print 'Aciertos totales: ', sum(correctness),
    print '(', '{0:,.2f}'.format(sum(correctness) * 100 / len(correctness)), '%)'
    # print 'Porcentaje ponderado: ', '{0:,.2f}'.format(ponderado * 100 / 4), '%'

wfdb = gd.WFDB()
wfdb.plot_train_cases()
trainingset, training_classes = wfdb.get_training_set()
# data_set, data_classes = wfdb.get_heartbeats()

test_set = wfdb.get_equally_prob_hb(100)
test_set = np.transpose(test_set)
test_classes = [i for i in range(0, 4) for j in range(0, 100)]
# SOM
# som = som.SOM(trainingset, length=trainingset.shape[1], epochs=7000, x=8, y=8)
# som.plot_weights_trains(trainingset, training_classes)
# som.plot_weigths(test_set, test_classes)
# guess = som.get_results(test_set)
# import pdb; pdb.set_trace()

# create the network
settings = {
    # Required settings
    "n_inputs": 32,                                              # Number of network input signals
    "layers": [(16, sigmoid_function),
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
dataset = [Instance(test_set[i, :], expected_output[test_classes[i]]) for i in range(0, len(test_set))]
# Instance( [inputs], [outputs] )
# dataset = [
#     Instance([0, 0], [0]), Instance([1, 0], [1]), Instance([0, 1], [1]), Instance([1, 1], [0])
# ]
# import pdb; pdb.set_trace()
training_set = [Instance(trainingset[i, :], expected_output[training_classes[i]]) for i in range(0, len(trainingset))]
# training_set = dataset
test_set = dataset
cost_function = sum_squared_error
print 'Starting to train...'
backpropagation(
    # Required parameters
    network,                     # the neural network instance to train
    training_set,                # the training dataset
    test_set,                    # the test dataset
    cost_function,               # the cost function to optimize

    # Optional parameters
    ERROR_LIMIT=1e-3,           # Error tolerance when terminating the learning
    max_iterations=10000,          # Regardless of the achieved error, terminate after max_iterations epochs. Default: infinite
    batch_size=0,               # Set the batch size. 0 implies using the entire training_set as a batch, 1 equals no batch learning, and any other number dictate the batch size
    input_layer_dropout=0.0,    # Dropout fraction of the input layer
    hidden_layer_dropout=0.0,   # Dropout fraction of in the hidden layer(s)
    print_rate=1000,            # The epoch interval to print progression statistics
    save_trained_network=False  # Whether to ask the user if they would like to save the network after training
)

# prediction_set = [Instance([0, 1]), Instance([1, 0])]
# data_set, data_classes = wfdb.get_organized()
# data_set = np.transpose(data_set)
# data_set = data_set[58320:60320, :]
# data_classes = data_classes[58320:60320]
num_data = 100
data_set = np.transpose(wfdb.get_equally_prob_hb(num_data))
data_classes = [i for i in range(0, 4) for _ in range(0, num_data)]
# import pdb; pdb.set_trace()
prediction_set = [Instance(data_set[i, :]) for i in range(0, len(data_set))]
# prediction_set = [Instance(trainingset[i, :]) for i in range(0, len(trainingset))]
guess_probs = np.round(network.predict(prediction_set)*100)
# print guess_probs
guess = [max(enumerate(g), key=lambda x: x[1])[0] for g in guess_probs]
# print guess
print_results(data_classes, guess)
# print_results(data_classes, guess)
# import pdb; pdb.set_trace()

