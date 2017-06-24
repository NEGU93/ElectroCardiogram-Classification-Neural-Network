import wfdb
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from pca import PCA
import som
# from matplotlib.mlab import PCA
# from sklearn.decomposition import PCA
from IPython.display import display


class WFDB:
    def __init__(self):
        # Import data from dataset 14172
        signals, fields = wfdb.srdsamp('../data/14172', channels=[0, 1])
        # print(self.signals)
        # display(self.fields)
        self.annotation = wfdb.rdann('../data/14172', 'atr')     # Import annotations from the same data set
        peaks = self.annotation.annsamp          # Let's find the self.peaks of each ECG where the annotations are
        signals = np.matrix.transpose(signals)

        # Don't take into account border self.peaks in case they are interrupted
        peaks[-1] = 0
        peaks[-2] = 0
        peaks[0] = 0
        peaks[1] = 0

        self.size_hb = 125                                                 # Size of heartbeat
        self.num_hb = sum(peaks > self.size_hb)                         # Count number of annotations
        self.heartbeats = np.zeros([self.num_hb, 2*self.size_hb])          # Save space for the data matrix
        idx_hb = 0                                          # Initialize the index for each peak
        for i in peaks:
            if i > self.size_hb:                                 # just in case the first heartbeat is truncated
                self.heartbeats[idx_hb] = signals[0][i - self.size_hb: i + self.size_hb]   # take all samples from left and right of i
                idx_hb += 1
        self.heartbeats = np.transpose(self.heartbeats)

        self.num_hb = self.heartbeats.shape[1]
        print('Number of Heartbeats: ', self.num_hb)

        self.heartbeats = self.calculate_means()         # Delete the mean from the data.

        self.print_kinds()                                  # View the kinds of hearbeats
        self.classes = self.calculate_classes()             # Get the kind of each heartbeat (N, V, S, J or others)
        [self.heartbeatsN, self.heartbeatsV, self.heartbeatsS, self.heartbeatsJ] = self.classify()  # get idx by class

        self.filter_data()
        [self.heartbeatsN, self.heartbeatsV, self.heartbeatsS, self.heartbeatsJ] = self.classify()
        # Training Set
        self.num_training_cases = 50  # set number of training cases
        [self.trainN, self.trainV, self.trainS, self.trainJ] = self.generate_training_cases(self.num_training_cases)
        self.training_classes = [i for i in range(0, 4) for _ in range(0, self.num_training_cases)]
        self.training_set = np.concatenate((self.trainN, self.trainV, self.trainS, self.trainJ), axis=1)
        self.training_set = np.transpose(self.training_set)

    def print_kinds(self):
        # Prints stats about the kinds of heartbeats
        # What kinds of heartbeats there are?
        kinds = set(self.annotation.anntype)
        print(kinds)
        # How many from each kind?
        for i in kinds:
            print i,
            print': ',
            print self.annotation.anntype.count(i)
            # print(annotation.anntype.index(i))

    def calculate_classes(self):
        # Generate an array of classes that identifies each heartbeat by a number
        classes = list(self.annotation.anntype)
        # Because a discarded the first 2 and last 2 elements
        classes = classes[1:-1]
        classes = classes[1:-1]
        for i in range(0, len(classes)):
            if classes[i] == 'N':
                classes[i] = 0
            elif classes[i] == 'V':
                classes[i] = 1
            elif classes[i] == 'S':
                classes[i] = 2
            elif classes[i] == 'J':
                classes[i] = 3
            else:
                classes[i] = 4
        return classes

    def classify(self):
        # Create arrays each with a single class heartbeat
        indexN = [i for i, x in enumerate(self.classes) if x == 0]
        indexV = [i for i, x in enumerate(self.classes) if x == 1]
        indexS = [i for i, x in enumerate(self.classes) if x == 2]
        indexJ = [i for i, x in enumerate(self.classes) if x == 3]

        heartbeatsN = self.heartbeats[:, indexN]
        heartbeatsV = self.heartbeats[:, indexV]
        heartbeatsS = self.heartbeats[:, indexS]
        heartbeatsJ = self.heartbeats[:, indexJ]

        return heartbeatsN, heartbeatsV, heartbeatsS, heartbeatsJ

    def filter_data(self):
        generator, chunk = self.get_organized(100)
        pca = PCA(generator, dim=32, size_hb=250)
        self.heartbeats = pca.reduce_dimensions(self.heartbeats)
        # SOM
        # som = som.SOM(trainingset, length=trainingset.shape[1], epochs=7000, x=8, y=8)
        # som.plot_weights_trains(trainingset, training_classes)
        # som.plot_weigths(test_set, test_classes)
        # guess = som.get_results(test_set)
        # import pdb; pdb.set_trace()

    def calculate_means(self):
        means = np.zeros(self.num_hb)  # Compute the mean of each heartbeat
        desvs = np.zeros(self.num_hb)
        for i in range(0, self.size_hb):
            means[i] = np.mean(self.heartbeats[:, i])
            desvs[i] = np.std(self.heartbeats[:, i])
        for i in range(0, self.num_hb):
            self.heartbeats[:, i] = self.heartbeats[:, i] - np.ones(2*self.size_hb)*means[i]
            self.heartbeats[:, i] = self.heartbeats[:, i] / max(desvs[i], 1)
        return self.heartbeats

    def generate_training_cases(self, num_training_cases=5):
        indexN = [i for i, x in enumerate(self.classes) if x == 0]
        indexV = [i for i, x in enumerate(self.classes) if x == 1]
        indexS = [i for i, x in enumerate(self.classes) if x == 2]
        indexJ = [i for i, x in enumerate(self.classes) if x == 3]

        indexNtrain = [randrange(0, len(indexN)) for _ in range(0, self.num_training_cases)]
        indexVtrain = [randrange(0, len(indexV)) for _ in range(0, self.num_training_cases)]
        indexStrain = [randrange(0, len(indexS)) for _ in range(0, self.num_training_cases)]
        indexJtrain = [randrange(0, len(indexJ)) for _ in range(0, self.num_training_cases)]

        trainN = self.heartbeatsN[:, indexNtrain]
        trainV = self.heartbeatsV[:, indexVtrain]
        trainS = self.heartbeatsS[:, indexStrain]
        trainJ = self.heartbeatsJ[:, indexJtrain]

        return trainN, trainV, trainS, trainJ

    # PLOT
    def plot_train_cases(self):
        plt.plot(self.trainN)
        plt.title('Normales')
        plt.savefig('../out/trainN.png')
        plt.close()
        plt.plot(self.trainV)
        plt.title('Ventriculares')
        plt.savefig('../out/trainV.png')
        plt.close()
        plt.plot(self.trainS)
        plt.title('Supraventriculares')
        plt.savefig('../out/trainS.png')
        plt.close()
        plt.plot(self.trainJ)
        plt.title('Nodales prematuros')
        plt.savefig('../out/trainJ.png')
        plt.close()

    def plot_each_case(self):
        plt.plot(self.heartbeatsN)
        plt.title('Normales')
        plt.savefig('../out/hearbeatsN.png')
        plt.close()
        plt.plot(self.heartbeatsV)
        plt.title('Ventriculares')
        plt.savefig('../out/hearbeatsV.png')
        plt.close()
        plt.plot(self.heartbeatsS)
        plt.title('Supraventriculares')
        plt.savefig('../out/hearbeatsS.png')
        plt.close()
        plt.plot(self.heartbeatsJ)
        plt.title('Nodales prematuros')
        plt.savefig('../out/hearbeatsJ.png')
        plt.close()

    # GETTERS
    def get_leng_hb(self):
        return 2*self.size_hb

    def get_training_set(self):
        return self.training_set, self.training_classes

    def get_heartbeats(self):
        return np.transpose(self.heartbeats), self.classes

    def get_organized(self, num_max=None):
        # returns the data in order (N, V, S, J) and of a fixed size of each one.
        # If the parameter is not given then it gives everything
        if num_max is None:
            num_max = max(self.heartbeatsN.shape[1],
                          self.heartbeatsV.shape[1],
                          self.heartbeatsS.shape[1],
                          self.heartbeatsJ.shape[1])
        indexN = np.zeros(min(self.heartbeatsN.shape[1], num_max))
        indexV = np.full(min(self.heartbeatsV.shape[1], num_max), 1)
        indexS = np.full(min(self.heartbeatsS.shape[1], num_max), 2)
        indexJ = np.full(min(self.heartbeatsJ.shape[1], num_max), 3)
        # import pdb; pdb.set_trace()
        return np.concatenate((self.heartbeatsN[:, 0:len(indexN)],
                               self.heartbeatsV[:, 0:len(indexV)],
                               self.heartbeatsS[:, 0:len(indexS)],
                               self.heartbeatsJ[:, 0:len(indexJ)]), axis=1), \
               np.concatenate((indexN, indexV, indexS, indexJ)).astype(int)
