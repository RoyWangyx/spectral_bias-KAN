import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.set_default_dtype(torch.float64)
from kan import *
import sys
from argparse import Namespace
from functools import reduce

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


dimen=1

def make_phased_waves(opt):
    t = np.arange(0, 1, 1./opt.N)
    if opt.A is None:
        yt = reduce(lambda a, b: a + b, 
                    [np.sin(2 * np.pi * ki * t + 2 * np.pi * phi) for ki, phi in zip(opt.K, opt.PHI)])
    else:
        yt = reduce(lambda a, b: a + b, 
                    [Ai * np.sin(2 * np.pi * ki * t + 2 * np.pi * phi) for ki, Ai, phi in zip(opt.K, opt.A, opt.PHI)])
    return t, yt

def fft(opt, yt):
    n = len(yt) # length of the signal
    k = np.arange(n)
    T = n/opt.N
    frq = k/T # two sides frequency range
    frq = frq[range(n//2)] # one side frequency range
    # -------------
    FFTYT = np.fft.fft(yt)/n # fft computing and normalization
    FFTYT = FFTYT[range(n//2)]
    fftyt = abs(FFTYT)
    return frq, fftyt

def to_torch_dataset_1d(opt, t, yt):
    t = torch.from_numpy(t).view(-1, 1)
    yt = torch.from_numpy(yt).view(-1, 1)
    if opt.CUDA:
        t = t.cuda()
        yt = yt.cuda()
    return t, yt

def make_model(opt): #mlp model
    layers = []
    layers.append(nn.Linear(opt.INP_DIM, opt.WIDTH))
    layers.append(nn.ReLU())
    for _ in range(opt.DEPTH - 2): 
        layers.append(nn.Linear(opt.WIDTH, opt.WIDTH))
        layers.append(nn.ReLU())
    layers.extend([nn.Linear(opt.WIDTH, opt.OUT_DIM)])
    model = nn.Sequential(*layers)
    if opt.CUDA:
        model = model.cuda()
    return model

def train_model(opt, model, input_, target, update_grid=True, grid_update_num=10, stop_grid_update_step=50, model_type='MLP'):
    # Build loss
    loss_fn = nn.MSELoss()
    # Build optim
    optim = torch.optim.Adam(model.parameters(), lr=opt.LR)
    #update grid
    grid_update_freq = int(stop_grid_update_step / grid_update_num)
    # Rec
    frames = []
    #model.train()
    # To cuda
    if opt.CUDA:
        input_ = input_.cuda()
        target = target.cuda()
    # Loop! 
    for iter_num in range(opt.NUM_ITER):
        if iter_num % (opt.NUM_ITER // 100) == 0: 
            print(">", end='')
        x = input_
        yt = target.view(-1, opt.OUT_DIM)
        if iter_num % grid_update_freq == 0 and iter_num < stop_grid_update_step and update_grid and model_type=='KAN':
            model.update_grid_from_samples(x)
        optim.zero_grad()
        y = model(x)
        loss = loss_fn(y, yt)
        loss.backward()
        optim.step()
        if iter_num % opt.REC_FRQ == 0: 
            # Measure spectral norm
            frames.append(Namespace(iter_num=iter_num, 
                                    prediction=y.data.cpu().numpy(), 
                                    loss=loss.item(), ))
                                    #spectral_norms=spectral_norm(model)))
    # Done
    #model.eval()
    return frames

def compute_spectra(opt, frames): 
    # Make array for heatmap
    dynamics = []
    xticks = []
    for iframe, frame in enumerate(frames): 
        # Compute fft of prediction
        frq, yfft = fft(opt, frame.prediction.squeeze())
        dynamics.append(yfft)
        xticks.append(frame.iter_num)
    return np.array(frq), np.array(dynamics), np.array(xticks)

def plot_spectral_dynamics(opt, all_frames, para,model_type):
    all_dynamics = []
    # Compute spectra for all frames
    for frames in all_frames: 
        frq, dynamics, xticks = compute_spectra(opt, frames)
        all_dynamics.append(dynamics)
    # Average dynamics over multiple frames
    # mean_dynamics.shape = (num_iterations, num_frequencies)
    mean_dynamics = np.array(all_dynamics).mean(0)
    # Select the frequencies which are present in the target spectrum
    freq_selected = mean_dynamics[:, np.sum(frq.reshape(-1, 1) == np.array(opt.K).reshape(1, -1), 
                                            axis=-1, dtype='bool')]
    # Normalize by the amplitude. Remember to account for the fact that the measured spectra 
    # are single-sided (positive freqs), so multiply by 2 accordingly
    norm_dynamics = 2 * freq_selected / np.array(opt.A).reshape(1, -1)
    # Plot heatmap
    plt.figure(figsize=(7, 6))
    plt.title(f'Model{model_type} Width{para[1]} Depth{para[0]}')
    sns.heatmap(norm_dynamics[::-1], 
                xticklabels=opt.K, 
                yticklabels=[(frame.iter_num if frame.iter_num % 10000 == 0 else '') 
                             for _, frame in zip(range(norm_dynamics.shape[0]), frames)][::-1], 
                vmin=0., vmax=1., 
                cmap=sns.cubehelix_palette(8, start=.5, rot=-.75, reverse=True, as_cmap=True))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Training Iteration")
    plt.savefig(f'width{para[1]}depth{para[0]}grid{para[2]}model{model_type}_inc.png')
    plt.show()

opt = Namespace()

# Data Generation
opt.N = 200
opt.K = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
opt.A = [1 for _ in opt.K]
opt.PHI = [np.random.rand() for _ in opt.K]
# Model parameters
opt.INP_DIM = 1
opt.OUT_DIM = 1

# Training
opt.CUDA = torch.cuda.is_available()
opt.NUM_ITER = 80000
opt.REC_FRQ = 100
opt.LR = 0.0003
opt.A = [0.1 * (a + 1) for a in range(len(opt.K))]

def go(opt, repeats=10,  model_type='KAN'):
    all_frames = []
    for _ in range(repeats): 
        # Sample random phase
        opt.PHI = [np.random.rand() for _ in opt.K]
        # Generate data
        x, y = to_torch_dataset_1d(opt, *make_phased_waves(opt))
        # Make model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_type == 'KAN':
            model = KAN(width=np.array([1] + [opt.WIDTH]*(opt.DEPTH-1) + [1]).tolist(), grid=opt.GRID, k=3).to(device)
        else:
            model = make_model(opt)
        # Train
        frames = train_model(opt, model, x, y,model_type)
        all_frames.append(frames)
        print('', end='\n')
    return all_frames
def run():

    # Grab the arguments that are passed in
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
    
    #depths = [2,3,4]
    depths = [10,6]
    #widths = [3,10]
    widths = [256,128,64,32]
    grids = [5]
    dims = [1]

    xx, yy, zz, ww = np.meshgrid(depths, widths, grids, dims)
    params_ = np.transpose(np.array([xx.reshape(-1,), yy.reshape(-1,), zz.reshape(-1,), ww.reshape(-1,)]))
    
    indices = np.arange(params_.shape[0])
    
    my_indices = indices[my_task_id:indices.shape[0]:num_tasks]

    for i in my_indices:
        opt.WIDTH = params_[i][1].astype('int')
        opt.DEPTH = params_[i][0].astype('int')
        opt.GRID = params_[i][2].astype('int')
        eq_amp_frames = go(opt, model_type='MLP')
        plot_spectral_dynamics(opt, eq_amp_frames, params_[i],'MLP')
run()