#Genero y entreno una red SOM con los quince latidos de entrenamiento
from minisom import MiniSom
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from pylab import plot, axis, show, pcolor, colorbar, bone


class SOM:
    def __init__(self, heartbeats, sigma=5, eta=0.5, x=8, y=8, length=250, epochs=50000):
        # Initialization and training #
        self.X = x
        self.Y = y

        self.som2 = MiniSom(self.X, self.Y, length, sigma=sigma, learning_rate=eta)

        self.som2.random_weights_init(heartbeats)

        print('Start training...')
        self.som2.train_random(heartbeats, epochs)
        print('Done training!')

    def plot_visualize_weights(self):
        fig = plt.figure()

        gs = gridspec.GridSpec(self.X, self.Y)

        for i in range(0, self.X):
            for j in range(0, self.Y):
                ax1 = fig.add_subplot(gs[i, j])
                ax1.plot(self.som2.weights[j, self.X - 1 - i, :], '-', c='k')

        gs.update(wspace=0.5, hspace=0.5)
        fig = plt.gcf()
        plt.savefig('../out/som_visual_weights.png')

    def plot_weights_trains(self, heartbeats, heartbeatsClasses):
        markers = ['o', 's', 'D', '^']
        colors = ['r', 'g', 'b', 'y']
        # red = N
        # green = V
        # blue = S
        # yellow = J
        fig, ax = plt.subplots()
        for cnt, xx in enumerate(heartbeats):
            # import pdb; pdb.set_trace()
            w = self.som2.winner(xx)  # getting the winner
            # palce a marker on the winning position for the sample xx
            plot(w[0] + .5 + np.random.randn(1) / 20, w[1] + .5 + np.random.randn(1) / 20, markers[heartbeatsClasses[cnt]],
                 markerfacecolor='None', markeredgecolor=colors[heartbeatsClasses[cnt]], markersize=8, markeredgewidth=2)
        axis([0, self.som2.weights.shape[0], 0, self.som2.weights.shape[1]])
        plt.title('red = N, green = V, blue = S, yellow = J')
        # x = np.linspace(0, self.X, 40)
        # yg = x - 3
        # ax.fill_between(x, 0, yg, facecolor='green', alpha=0.5)
        # yb = x + 1
        # ax.fill_between(x, yg, yb, facecolor='blue', alpha=0.5)
        # x2 = np.linspace(0, 6, 30)
        # ax.fill_between(x2, 6, 8, facecolor='yellow', alpha=0.5)
        plt.savefig('../out/som_weights_train.png')

    def plot_weigths(self, heartbeats, heartbeatsClasses):
        markers = ['o', 's', 'D', '^', '*']
        colors = ['r', 'g', 'b', 'y', 'k']
        # red = N
        # green = V
        # blue = S
        # yellow = J
        # import pdb; pdb.set_trace()
        for cnt, xx in enumerate(heartbeats):
            # import pdb; pdb.set_trace()
            w = self.som2.winner(xx)  # getting the winner
            # palce a marker on the winning position for the sample xx
            plot(w[0] + .5 + np.random.randn(1) / 20, w[1] + .5 + np.random.randn(1) / 20,
                 markers[heartbeatsClasses[cnt]],
                 markerfacecolor='None',
                 markeredgecolor=colors[heartbeatsClasses[cnt]], markersize=8, markeredgewidth=2)
        axis([0, self.som2.weights.shape[0], 0, self.som2.weights.shape[1]])
        plt.title('red = N, green = V, blue = S, yellow = J')
        plt.savefig('../out/som_weights.png')

    def print_results(self, heartbeats, heartbeatsClasses):
        N_s = 0
        N_j = 0
        N_v = 0
        N_n = 0
        N_n_right = 0
        N_n_wrong = 0
        N_s_right = 0
        N_s_wrong = 0
        N_v_right = 0
        N_v_wrong = 0
        N_j_right = 0
        N_j_wrong = 0
        N_none = 0
        N_none_right = 0
        N_none_wrong = 0

        for cnt, xx in enumerate(heartbeats):
            w = self.som2.winner(xx)  # getting the winner
            if w[0] > w[1] + 3:
                N_v += 1
                if heartbeatsClasses[cnt] == 1:
                    N_v_right += 1
                else:
                    N_v_wrong += 1
            elif (w[0] < 3 and w[1] < 3) or (w[1] < w[0] + 1):
                N_s += 1
                if heartbeatsClasses[cnt] == 2:
                    N_s_right += 1
                else:
                    N_s_wrong += 1
            elif (w[0] > 6) and (w[1] < 5):
                N_j += 1
                if heartbeatsClasses[cnt] == 3:
                    N_j_right += 1
                else:
                    N_j_wrong += 1
            elif w[1] > w[0]:
                N_n += 1
                if heartbeatsClasses[cnt] == 0:
                    N_n_right += 1
                else:
                    N_n_wrong += 1
            else:
                N_none += 1
                if heartbeatsClasses[cnt] == 4:
                    N_none_right += 1
                else:
                    N_none_wrong += 1

        print('Normales hallados: ')
        print(N_n)
        print('Correctamente: ')
        print(N_n_right)
        print('Incorrectamente: ')
        print(N_n_wrong)
        print('Porcentaje correcto: ')
        print('{0:,.2f}'.format(N_n_right * 100 / max(N_n, 1)))
        print('------------------------------')
        print('Ventriculares hallados: ')
        print(N_v)
        print('Correctamente: ')
        print(N_v_right)
        print('Incorrectamente: ')
        print(N_v_wrong)
        print('Porcentaje correcto: ')
        print('{0:,.2f}'.format(N_v_right * 100 / max(N_v, 1)))
        print('------------------------------')
        print('Supraventriculares hallados: ')
        print(N_s)
        print('Correctamente: ')
        print(N_s_right)
        print('Incorrectamente: ')
        print(N_s_wrong)
        print('Porcentaje correcto: ')
        print('{0:,.2f}'.format(N_s_right * 100 / max(N_s, 1)))
        print('------------------------------')
        print('Nodales prematuros hallados: ')
        print(N_j)
        print('Correctamente: ')
        print(N_j_right)
        print('Incorrectamente: ')
        print(N_j_wrong)
        print('Porcentaje correcto: ')
        print('{0:,.2f}'.format(N_j_right * 100 / max(N_j, 1)))
        print('------------------------------')
        print('Otros hallados: ')
        print(N_none)
        print('Correctamente: ')
        print(N_none_right)
        print('Incorrectamente: ')
        print(N_none_wrong)
        print('Porcentaje correcto: ')
        print('{0:,.2f}'.format(N_none_right * 100 / max(N_none, 1)))

    def get_results(self, heartbeats):
        # import pdb; pdb.set_trace()
        guessed_classes = np.zeros(heartbeats.shape[0])
        for cnt, xx in enumerate(heartbeats):
            w = self.som2.winner(xx)  # getting the winner
            if w[0] > w[1] + 3:
                guessed_classes[cnt] = 1
            elif (w[0] < 3 and w[1] < 3) or (w[1] < w[0] + 1):
                guessed_classes[cnt] = 2
            elif (w[0] > 6) and (w[1] < 5):
                guessed_classes[cnt] = 3
            elif w[1] > w[0]:
                guessed_classes[cnt] = 0
            else:
                guessed_classes[cnt] = 4
        return guessed_classes
