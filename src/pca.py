import numpy as np


def pca(data, dim=32, size_hb=250):
    C = np.cov(data)
    import pdb; pdb.set_trace()
    (auval, auvec) = np.linalg.eigh(C)

    # Sort from bigger to smaller
    idx = auval.argsor()[::-1]
    auval = auval[idx]
    auvec = auvec[:, idx]
    auvec = np.matrix.transpose(auvec)

    u = np.dot(auvec, data)     # transform to PCA space
    u[dim:size_hb] = 0          # delete 'extra' dimensions

    recover = np.dot(np.matrix.transpose(auvec), u)
    return recover
