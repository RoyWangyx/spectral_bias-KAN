# deep ritz method favors kans
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.set_default_dtype(torch.float64)
from torch.autograd import grad
from kan import *
import sys
from argparse import Namespace
from functools import reduce

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
class MLP(nn.Module):
    def __init__(self, width, depth,dimen):
        super(MLP, self).__init__()
        self.width = width
        if depth < 2:
          raise ValueError(f"depth must be at least 2, got {depth}")
        self.depth = depth
        self.act = nn.Tanh()

        # layer definitions
        self.FC_dict = nn.ModuleDict()
        self.FC_dict[f"layer 1"] = nn.Linear(dimen, self.width)
        for i in range(2, depth):
          self.FC_dict[f"layer {i}"] = nn.Linear(self.width, self.width)
        self.FC_dict[f"layer {depth}"] = nn.Linear(self.width, 1)

    def forward(self, x):
      for key, layer in self.FC_dict.items():
        x = layer(x)
        if key != f"layer {self.depth}":
          x = self.act(x)
      return x
def run():

    # Grab the arguments that are passed in
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
    model_type = ['KAN','MLP',"KAN3"]
    scales = [1,2,4,8,16,32]
    begin = [0,-1]
    xx, yy,zz = np.meshgrid(model_type, scales,begin)
    params_ = np.transpose(np.array([xx.reshape(-1,), yy.reshape(-1,), zz.reshape(-1,)]))
    
    indices = np.arange(params_.shape[0])
    
    my_indices = indices[my_task_id:indices.shape[0]:num_tasks]

    for i in my_indices:
        model_choice = params_[i][0]
        grids = [20,40]#[5,10,15,20,25]
        freq = params_[i][1].astype('int')
        #if freq >10:
        #    for i in range(len(grids)):
        #        grids[i] =2*grids[i]        

        
# initialize a model; training points are grid-based
        torch.manual_seed(0)
        np.random.seed(0)
        dimen = 1
        loss_choice = "PINN"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_choice == "MLP":
            WIDTH = 256
            DEPTH = 6
            model = MLP(width=WIDTH, depth=DEPTH, dimen=dimen).to(device)
        else:
            WIDTH = 10
            DEPTH = 2
            if model_choice == "KAN3":
                DEPTH = 3
            kan_size = [dimen] + [WIDTH]*(DEPTH-1) + [1]
            model = KAN(width=kan_size, grid=grids[0], k=3,noise_scale=0.1, symbolic_enabled=False).to(device)


        batchsize = 2000 # batchsize for interior points per direction
        batchsize_b = 1
        update_grid_freq = 5
        stop_update_grid_step = 50
        step = 10000
        begin_point = params_[i][2].astype('int')
        end_point  = 1


        true_losses_l2 = []
        true_losses_h1 = []
        Ai = 0.01

        w = torch.linspace(begin_point, end_point, steps=batchsize,device=device, requires_grad=True).to(device).view(-1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for grid in grids:
            print('grid =',grid)
           
            for ep in range(step):
                # 2d grid points
                
                optimizer.zero_grad()
                if ep==0 and model_choice != "MLP":
                    model2 = KAN(width=kan_size, grid=grid, k=3, noise_scale=0.1, symbolic_enabled=False).to(device)
                    if grid != grids[0]:
                        model2.initialize_from_another_model(model, w)
                        model = model2
                        model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                if ep % update_grid_freq == 0 and ep <= stop_update_grid_step and model_choice != "MLP":
                    model.update_grid_from_samples(w)
                
                
                # evaluation

                U = model(w)
                U1 = grad(U.sum(), w, create_graph=True)[0]
                U11 = grad(U1.sum(), w, create_graph=True)[0]
                #interior and smoothness losses
                if loss_choice == "DRM":
                    Fu = 1/2*U1*U1 -U*(freq*torch.sin(freq*np.pi*w)+torch.sin(np.pi*w))*np.pi**2
                else:
                    Fu = (U11 +(freq*torch.sin(freq*np.pi*w)+torch.sin(np.pi*w))*np.pi**2)**2
                loss_i = torch.sum(Fu)/batchsize

                w1 = begin_point*torch.ones(1, device=device, requires_grad=True).to(device).view(-1, 1) # (batch, 1)
                w2 = end_point*torch.ones(1, device=device, requires_grad=True).to(device).view(-1, 1)
            
            
                U_1 = model(w1)
                U_2 = model(w2)


                loss_b = torch.norm(U_1)**2/batchsize_b + torch.norm(U_2)**2/batchsize_b
                loss = Ai*loss_i +1*loss_b 
                loss.backward()
                optimizer.step()
                # true loss
                Utruth = torch.sin(freq*np.pi*w)/freq+torch.sin(np.pi*w)
                loss_truth = torch.norm(Utruth - U)**2/batchsize
                
                U1_t = grad(Utruth.sum(), w, create_graph=True)[0]
                loss_truth_h1 = torch.norm(U1_t - U1)**2/batchsize

                true_losses_l2.append(loss_truth.item())
                true_losses_h1.append(loss_truth_h1.item())
                if ep % 10 == 0:
                    print("Epoch is {}, interior loss is {}, boundary loss is {},  overall loss is {}, l2 true loss is {}, h1 true loss is {}".format(ep, loss_i.item(),loss_b.item(), loss.item(),loss_truth.item(),loss_truth_h1.item()))
                    #plt.plot(Utruth.detach().cpu().numpy(),label='true')    
                    #plt.plot(U.detach().cpu().numpy(),label='pred')
                    #plt.show()
        np.savetxt(f'l2adam{begin_point}_scale_{freq}model_{model_choice}.txt', np.array(true_losses_l2).flatten().reshape(-1,1))
        np.savetxt(f'h1adam{begin_point}_scale_{freq}model_{model_choice}.txt', np.array(true_losses_h1).flatten().reshape(-1,1))
run()