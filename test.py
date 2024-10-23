import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg as la
from scipy.stats._mstats_basic import winsorize
from scipy.linalg import orth
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

seed = 5
np.random.seed(seed)
# M * N
def construct_simulation_data(times):
    exp = pd.read_csv("./ASEFM/data/real_data/GSE_exp.csv",sep=',').dropna()
    methy = pd.read_csv("./ASEFM/data/real_data/GSE_methy.csv",sep=',').dropna()
    mirna = pd.read_csv("./ASEFM/data/real_data/GSE_mirna.csv",sep=',').dropna()
    n_sample = 400
    num_subspaces = 5
    subspace_dim = 5
    num_points_per_subspace = int(n_sample / num_subspaces)

    sub_exp = exp.sample(n=n_sample,axis=1).to_numpy().T
    sub_methy = methy.sample(n=n_sample,axis=1).to_numpy().T
    sub_mirna = mirna.sample(n=n_sample,axis=1).to_numpy().T

    simdata = []
    for omics in [sub_exp, sub_methy, sub_mirna]:
        ambient_dim = omics.shape[1]
        noise_level = 0.3
        label = np.empty(num_points_per_subspace * num_subspaces, dtype=int)
        data = np.empty((num_points_per_subspace * num_subspaces, ambient_dim))
        for i in range(num_subspaces):
            basis = np.random.normal(size=(ambient_dim, subspace_dim))
            basis = orth(basis)
            base_index = i * num_points_per_subspace
            now_index = num_points_per_subspace + base_index
            pca = PCA(n_components=subspace_dim)

            omics_transform = pca.fit_transform(omics[base_index:now_index, ])
            data_per_subspace = np.matmul(basis, omics_transform.T).T

            data[base_index:now_index, ] = data_per_subspace
            label[base_index:now_index,] = i
        simdata.append(data)
        print(data.shape)

        data += np.random.normal(size=(num_points_per_subspace * num_subspaces, ambient_dim)) * noise_level
    return simdata, label

simdata, label = construct_simulation_data(0)
np.savetxt('./ASEFM/data/simdata3/methy_0.csv', simdata[1], delimiter=',')
