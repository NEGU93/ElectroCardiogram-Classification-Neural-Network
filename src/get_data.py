import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pca
# from matplotlib.mlab import PCA
from sklearn.decomposition import PCA
from IPython.display import display


class WFDB:
    def __init__(self):
        # Import data from dataset 14172
        self.signals, self.fields = wfdb.srdsamp('../data/14172', channels=[0, 1])
        # print(self.signals)
        # display(self.fields)
        self.annotation = wfdb.rdann('../data/14172', 'atr')     # Import annotations from the same data set
        self.peaks = self.annotation.annsamp                          # Let's find the self.peaks of each ECG where the annotations are
        self.signals = np.matrix.transpose(self.signals)

        # Don't take into account border self.peaks in case they are interrupted
        self.peaks[-1] = 0
        self.peaks[-2] = 0
        self.peaks[0] = 0
        self.peaks[1] = 0

        self.size_hb = 125                                                 # Size of heartbeat
        self.num_hb = sum(self.peaks > self.size_hb)                         # Count number of annotatios
        self.heartbeats = np.zeros([self.num_hb, 2*self.size_hb])          # Save space for the data matrix
        self.norm_heartbeats = np.zeros([self.num_hb, 2*self.size_hb])     # Save space for the normalized data matrix
        idx_hb = 0                                          # Initialize the index for each peak
        for i in self.peaks:
            if i > self.size_hb:                                 # just in case the first heartbeat is truncated
                self.heartbeats[idx_hb] = self.signals[0][i - self.size_hb: i + self.size_hb]   # take all samples from left and right of i
                idx_hb += 1

        self.heartbeats = np.transpose(self.heartbeats)

        self.num_hb = self.heartbeats.shape[1]
        print('Number of Heartbeats: ', self.num_hb)

        self.means, self.desv = self.calculate_means()         # Delete the mean from the data.
        for i in range(0, self.num_hb):
            self.heartbeats[:, i] = self.heartbeats[:, i] - np.ones(2*self.size_hb)*self.means[i]
            self.heartbeats[:, i] = self.heartbeats[:, i] / max(self.desv[i], 1)

        self.filter_data()

        self.view_kinds()                               # View the kinds of hearbeats
        self.classes = self.calculate_classes()         # Get the kind of each heartbeat (N, V, S, J or others)
        [self.heartbeatsN, self.heartbeatsV, self.heartbeatsS, self.heartbeatsJ] = self.classify()  # get idx by class

        # Training Set
        self.num_training_cases = 20             # set number of training cases
        [self.trainN, self.trainV, self.trainS, self.trainJ] = self.get_training_cases(self.num_training_cases)
        self.training_classes = [i for i in range(0, 4) for j in range(0, self.num_training_cases)]
        self.training_set = np.concatenate((self.trainN, self.trainV, self.trainS, self.trainJ), axis=1)
        self.training_set = np.transpose(self.training_set)

    def get_training_cases(self, num_training_cases=5):
        trainN = self.heartbeatsN[:, range(0, num_training_cases)]
        trainV = self.heartbeatsV[:, range(0, num_training_cases)]
        trainS = self.heartbeatsS[:, range(0, num_training_cases)]
        trainJ = self.heartbeatsJ[:, range(0, num_training_cases)]

        return trainN, trainV, trainS, trainJ

    def get_heartbeats(self):
        return np.transpose(self.heartbeats), self.classes

    def get_equally_prob_hb(self, num):
        return np.concatenate((self.heartbeatsN[:, self.num_training_cases:self.num_training_cases + num],
                               self.heartbeatsV[:, self.num_training_cases:self.num_training_cases + num],
                               self.heartbeatsS[:, self.num_training_cases:self.num_training_cases + num],
                               self.heartbeatsJ[:, self.num_training_cases:self.num_training_cases + num]),
                              axis=1)

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
        self.heartbeats = np.transpose(self.heartbeats)
        # import pdb; pdb.set_trace()
        # for i in range(0, self.heartbeats.shape[0]):
        #    self.heartbeats[i, :] = pca.pca(self.heartbeats[i, :])
        # results = PCA(self.heartbeats)
        # self.heartbeats = PCA(n_components=64).fit(self.heartbeats).transform(self.heartbeats)

        self.heartbeats = np.transpose(self.heartbeats)

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

    def view_kinds(self):
        # What kinds of heartbeats there are?
        kinds = set(self.annotation.anntype)
        print(kinds)
        # How many from each kind?
        for i in kinds:
            print i,
            print': ',
            print self.annotation.anntype.count(i)
            # print(annotation.anntype.index(i))

    def calculate_means(self):
        means = np.zeros(self.num_hb)  # Compute the mean of each heartbeat
        desvs = np.zeros(self.num_hb)
        for i in range(0, self.size_hb):
            means[i] = np.mean(self.heartbeats[:, i])
            desvs[i] = np.std(self.heartbeats[:, i])
        return means, desvs

    def get_means(self):
        return self.means

    def get_training_set(self):
        return self.training_set, self.training_classes

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

    def get_leng_hb(self):
        return 2*self.size_hb
