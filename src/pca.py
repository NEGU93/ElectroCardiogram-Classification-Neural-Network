import numpy as np


class PCA:
    def __init__(self, data, dim=32, size_hb=250):
        C = np.cov(data)
        self.dim = dim
        self.size_hb = size_hb
        # import pdb; pdb.set_trace()
        (self.auval, self.auvec) = np.linalg.eigh(C)

        # Sort from bigger to smaller
        idx = self.auval.argsort()[::-1]
        self.auval = self.auval[idx]
        self.auvec = self.auvec[:, idx]
        self.auvec = np.matrix.transpose(self.auvec)

    def reduce_dimensions(self, data):
        u = np.dot(self.auvec, data)  # transform to PCA space
        u[self.dim:self.size_hb] = 0  # delete 'extra' dimensions
        recover = np.dot(np.matrix.transpose(self.auvec), u)
        return u[0:self.dim]

    def recover_data(self, reduced_data):
        u = np.concatenate((reduced_data, np.zeros(self.size_hb - self.dim)))
        return np.dot(np.matrix.transpose(self.auvec), u)
