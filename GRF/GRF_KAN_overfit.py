import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.set_default_dtype(torch.float64)
from kan import *
from kan.utils import create_dataset
from torch.autograd import grad
from argparse import Namespace
from functools import reduce
from scipy.sparse.linalg import eigsh
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
import sys 
from scipy.spatial.distance import cdist
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rbf_kernel(X1, X2, length_scale):
    pairwise_dists = cdist(X1, X2, 'sqeuclidean')
    return np.exp(-0.5 * pairwise_dists / length_scale**2)

def make_dataset(length_scale, input_dim, n_points = 50000):
    torch.manual_seed(0)
    np.random.seed(0)
    # Define parameters
    n_samples = 1  # Number of function samples
    n_eigenfunctions = int(input_dim**3/length_scale**2)  # Number of eigenfunctions to use for approximation
    cutoff_scale = 0.1  # Fraction of eigenvalues to keep
    X = np.random.rand(n_points, input_dim) * 2 - 1  # Random points in [-1, 1]^input_dim

    # Compute covariance matrix
    K = rbf_kernel(X, X, length_scale)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = eigsh(K, k=n_eigenfunctions)
    #threshold of eigenvalue/eigenvalue_1<0.1
    n_eigenfunctions=len(eigenvalues[eigenvalues>cutoff_scale*eigenvalues[-1]])
    assert n_eigenfunctions < input_dim**3/length_scale**2
    eigenvalues, eigenvectors = eigsh(K, k=n_eigenfunctions)

    xi = np.random.normal(size=(n_eigenfunctions, n_samples))

    # Sample functions from the GP
    f_samples = np.zeros((n_points, n_samples))
    for i in range(n_samples):
        for j in range(n_eigenfunctions):
            f_samples[:, i] += np.sqrt(eigenvalues[j]) * eigenvectors[:, j] * xi[j, i]*np.sign(eigenvectors[0, j])


    train = 0.8
    num_train =int(train*n_points)
    normalization = np.std(f_samples[:num_train, :])
    dataset={}
    dataset['train_input'] = torch.from_numpy(X[:num_train,:]).to(device)
    dataset['test_input'] = torch.from_numpy(X[num_train:,:]).to(device)
    dataset['train_label'] = torch.from_numpy(f_samples[:num_train, :]).to(device)/normalization
    dataset['test_label'] = torch.from_numpy(f_samples[num_train:, :]).to(device)/normalization
    return dataset

def run():

    # Grab the arguments that are passed in
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
    model_choice = 'KAN'
    depths = [2,3,4]
    #depths = [6]
    widths = [10]
    #widths = [256]
    grids = [10,20,30,40,50]
    dims = [3]
    scales = [0.125,0.25,0.5,1]

    xx, yy, zz, ww = np.meshgrid(depths, widths,  dims,scales)
    params_ = np.transpose(np.array([xx.reshape(-1,), yy.reshape(-1,), zz.reshape(-1,), ww.reshape(-1,)]))
    
    indices = np.arange(params_.shape[0])
    
    my_indices = indices[my_task_id:indices.shape[0]:num_tasks]

    for i in my_indices:
        DIM = params_[i][2].astype('int')
        WIDTH = params_[i][1].astype('int')
        DEPTH = params_[i][0].astype('int')
        SCALE = params_[i][3]
        torch.manual_seed(0)
        np.random.seed(0)
        dataset=make_dataset(length_scale=SCALE, input_dim=DIM, n_points = 50000)
        test_losses = []
        train_losses = []
        model = KAN(width=np.array([DIM] + [WIDTH]*(DEPTH-1) + [1]).tolist(), grid=grids[0], k=3, symbolic_enabled=False, auto_save=True).to(device) 
        result = model.fit(dataset, opt="LBFGS", steps=100, start_grid_update_step=-1)
        train_losses.append(result['train_loss'])
        test_losses.append(result['test_loss'])
        for grid in grids[1:]:
            print('grid =',grid)
            model = model.refine(grid).to(device)
            print(model.act_fun[0].grid.device)
            result = model.fit(dataset, opt="LBFGS", steps=100, start_grid_update_step=-1)
            train_losses.append(result['train_loss'])
            test_losses.append(result['test_loss'])
            # save loss curves
        np.savetxt(f'trainloss_overfit_depth{DEPTH}_scale_{SCALE}_model_{model_choice}.txt', np.array(train_losses).flatten().reshape(-1,1))
        np.savetxt(f'testloss_overfit_depth{DEPTH}_scale_{SCALE}_model_{model_choice}.txt', np.array(test_losses).flatten().reshape(-1,1))
run()