import get_data as gd
import numpy as np
import sys
import time
from nimblenet.activation_functions import sigmoid_function
from nimblenet.neuralnet import NeuralNet
from nimblenet.data_structures import Instance
from nimblenet.learning_algorithms import backpropagation
from nimblenet.cost_functions import sum_squared_error

# Prepare redirect print file
orig_stdout = sys.stdout
# Parameters
hidden_layer_dropout = 0.8
input_layer_dropout = 0.8
early_stop = 30
num_training_cases = 10
data_reduction = 32
hidden_nodes = 24
num_data_set = 2000

# Create log file
es = '_no_es'
if early_stop:
    es = '_earlyStop' + str(early_stop)
if num_data_set is None:
    data_num = 'all'
else:
    data_num = str(num_data_set)
text_file = open(
    "../out/logs/" + str(time.time()) + "_" +
    data_num + "_training" + str(num_training_cases) +
    es +
    "_dropout" + str(hidden_layer_dropout) +
    "_reductionDim" + str(data_reduction) +
    ".txt", "w")
sys.stdout = text_file


def print_results(original, guess):
    correctness = np.equal(original, guess)
    cases = [(0, 'Normales'), (1, 'Ventriculares'), (2, 'Supraventriculares'), (3, 'Nodales Prematuros')]  # , (4, 'otros')]
    ponderado = 0.0
    enfermos_encontrados = 0.0
    falsos_negativos = 0.0
    falsos_positivos = 0.0
    for c, name in cases:
        count = 0.0
        total = 0.0
        for i in range(0, len(guess)):
            if c == guess[i]:
                total += 1.0
                if correctness[i]:          # Good diagnose
                    count += 1.0
        totales_reales = sum([1 for o in original if o == c])
        ponderado += (count * 100 / max(total, 1))
        if name is not 'Normales':
            enfermos_encontrados += 100 - ((totales_reales - count) * 100 / max(totales_reales, 1))
        else:
            falsos_negativos = (total - count)
            falsos_positivos = (totales_reales - count)
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
    # print 'Porcentaje ponderado: ', '{0:,.2f}'.format(ponderado / 4), '%'
    print 'Porcentaje enfermos ponderado: ', '{0:,.2f}'.format(enfermos_encontrados / 3), '%'
    print 'Falsos negativos: ', falsos_negativos
    print 'Falsos positivos: ', falsos_positivos

wfdb = gd.WFDB(num_training_cases, data_reduction)
wfdb.plot_train_cases()
trainingset, training_classes = wfdb.get_training_set()
testset, test_classes = wfdb.get_organized(100)
testset = np.transpose(testset)
expected_output = [
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
]
# Training the net
test_set = [Instance(testset[i, :], expected_output[test_classes[i]]) for i in range(0, len(testset))]
training_set = [Instance(trainingset[i, :], expected_output[training_classes[i]]) for i in range(0, len(trainingset))]

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
    max_iterations=(),                          # Regardless of the achieved error, terminate after max_iterations epochs. Default: infinite
    batch_size=0,                               # Set the batch size. 0 implies using the entire training_set as a batch, 1 equals no batch learning, and any other number dictate the batch size
    input_layer_dropout=input_layer_dropout,    # Dropout fraction of the input layer
    hidden_layer_dropout=hidden_layer_dropout,   # Dropout fraction of in the hidden layer(s)
    print_rate=1000,                            # The epoch interval to print progression statistics
    save_trained_network=False,                 # Whether to ask the user if they would like to save the network after training
    early_stop=early_stop
)

# prediction_set = [Instance([0, 1]), Instance([1, 0])]
data_set, data_classes = wfdb.get_organized(num_data_set)
data_set = np.transpose(data_set)
# import pdb; pdb.set_trace()
prediction_set = [Instance(data_set[i, :]) for i in range(0, len(data_set))]
# prediction_set = [Instance(trainingset[i, :]) for i in range(0, len(trainingset))]
guess_probs = np.round(network.predict(prediction_set)*100)
guess = [max(enumerate(g), key=lambda x: x[1])[0] for g in guess_probs]
print_results(data_classes, guess)
# import pdb; pdb.set_trace()
sys.stdout = orig_stdout
text_file.close()

