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

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, width, depth, dimen):
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
    batchsize = 5000 
    batchsize_b = 100 
    weight = 0.01  
    grids = [10,20,30,40,50]
    update_grid_freq = 5
    stop_update_grid_step = 50
    step = 100

    # Grab the arguments that are passed in
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
    
    depths = [2,3,4]
    frequencies = [3,6,9,12]
    #widths = [5,7,10]
    widths = [10] #[20,30,40,50]
    dims = [2,3]
       
    xx, yy, zz, ww = np.meshgrid(depths, widths, frequencies, dims)
    params_ = np.transpose(np.array([xx.reshape(-1,), yy.reshape(-1,), zz.reshape(-1,), ww.reshape(-1,)]))
    
    indices = np.arange(params_.shape[0])
    
    my_indices = indices[my_task_id:indices.shape[0]:num_tasks]

    for i in my_indices:
      
        model_choice = "KAN" # "MLP" or "KAN
        WIDTH = params_[i][1].astype('int')
        DEPTH = params_[i][0].astype('int')
        FREQUENCY = params_[i][2].astype('int')
        DIM = params_[i][3].astype('int')

        torch.manual_seed(0)
        np.random.seed(0)

        true_losses_l2 = []
        true_losses_h1 = []

        if model_choice == "MLP":
            model = MLP(dim = DIM, width=WIDTH, depth=DEPTH).to(device)
        else:
            model = KAN(width=np.array([DIM] + [WIDTH]*(DEPTH-1) + [1]).tolist(), grid=grids[0], k=3, symbolic_enabled=False, auto_save=True).to(device)
        if DIM ==3:
            w1 = torch.rand(batchsize, 1, device=device, requires_grad=True)
            w2 = torch.rand(batchsize, 1, device=device, requires_grad=True)
            w3 = torch.rand(batchsize, 1, device=device, requires_grad=True)
            y=torch.cat([w1,w2,w3], dim=-1).to(device)

            for grid in grids:
                print('grid =',grid)
                def closure():
                    optimizer.zero_grad()
                    U = model(y)
                    U1 = grad(U.sum(), w1, create_graph=True)[0]
                    U2 = grad(U.sum(), w2, create_graph=True)[0]
                    U3 = grad(U.sum(), w3, create_graph=True)[0]
                    U11 = grad(U1.sum(), w1, create_graph=True)[0]
                    U22 = grad(U2.sum(), w2, create_graph=True)[0]
                    U33 = grad(U3.sum(), w3, create_graph=True)[0]
                    #interior and smoothness losses
                    Fu = U22+U11+U33 +3*torch.sin(np.pi*w3)*torch.sin(np.pi*w2)*torch.sin(np.pi*w1)*np.pi**2+3*FREQUENCY**2*torch.sin(FREQUENCY*np.pi*w3)*torch.sin(FREQUENCY*np.pi*w2)*torch.sin(FREQUENCY*np.pi*w1)*np.pi**2
                    loss_i = torch.norm(Fu)**2/batchsize


                    #boundary loss
                    w1_b1 = 0*torch.ones(batchsize_b,1, device=device, requires_grad=True) # (batch, 1)
                    w1_b2 = 1*torch.ones(batchsize_b,1, device=device, requires_grad=True)
                    w2_1 = torch.rand(batchsize_b, 1, device=device, requires_grad=True)
                    w3_1 = torch.rand(batchsize_b, 1, device=device, requires_grad=True)
                    
                    z1=torch.cat([w1_b1,w2_1,w3_1], dim=-1).to(device)
                    z2=torch.cat([w1_b2,w2_1,w3_1], dim=-1).to(device)

                    U_1 = model(z1)
                    U_2 = model(z2)

                    w2_b1 = 0*torch.ones(batchsize_b,1, device=device, requires_grad=True) # (batch, 1)
                    w2_b2 = 1*torch.ones(batchsize_b, 1,device=device, requires_grad=True)
                    w1_2 = torch.rand(batchsize_b, 1, device=device, requires_grad=True)
                    w3_2 = torch.rand(batchsize_b, 1, device=device, requires_grad=True)
                    
                    z3=torch.cat([w1_2,w2_b1,w3_2], dim=-1).to(device)
                    z4=torch.cat([w1_2,w2_b2,w3_2], dim=-1).to(device)

                    U_3 = model(z3)
                    U_4 = model(z4)

                    w3_b1 = 0*torch.ones(batchsize_b,1, device=device, requires_grad=True) # (batch, 1)
                    w3_b2 = 1*torch.ones(batchsize_b, 1,device=device, requires_grad=True)
                    w1_3 = torch.rand(batchsize_b, 1, device=device, requires_grad=True)
                    w2_3 = torch.rand(batchsize_b, 1, device=device, requires_grad=True)
                    
                    z5=torch.cat([w1_3,w2_3,w3_b1], dim=-1).to(device)
                    z6=torch.cat([w1_3,w2_3,w3_b2], dim=-1).to(device)

                    U_5 = model(z5)
                    U_6 = model(z6)

                    loss_b = torch.norm(U_1)**2/batchsize_b + torch.norm(U_2)**2/batchsize_b+torch.norm(U_3)**2/batchsize_b + torch.norm(U_4)**2/batchsize_b+torch.norm(U_5)**2/batchsize_b + torch.norm(U_6)**2/batchsize_b
                    loss = weight*loss_i +1*loss_b 
            #back propogation of losses
                    loss.backward()
                    return loss

                for ep in range(step):
                    
                    optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

                    if ep==0 and model_choice == "KAN":
                        model2 = KAN(width=np.array([DIM] + [WIDTH]*(DEPTH-1) + [1]).tolist(), grid=grid, k=3, symbolic_enabled=False, auto_save=True).to(device)
                        if grid != grids[0]:
                            model2.initialize_from_another_model(model, y)
                            model = model2
                            model.to(device)
                        optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

                    if ep % update_grid_freq == 0 and ep <= stop_update_grid_step and model_choice == "KAN":
                        model.update_grid_from_samples(y)



                    optimizer.step(closure)


                    U = model(y)
                    U1 = grad(U.sum(), w1, create_graph=True)[0]
                    U2 = grad(U.sum(), w2, create_graph=True)[0]
                    U3 = grad(U.sum(), w3, create_graph=True)[0]
                    U11 = grad(U1.sum(), w1, create_graph=True)[0]
                    U22 = grad(U2.sum(), w2, create_graph=True)[0]
                    U33 = grad(U3.sum(), w3, create_graph=True)[0]
                    #interior and smoothness losses
                    Fu = U22+U11+U33 +3*torch.sin(np.pi*w3)*torch.sin(np.pi*w2)*torch.sin(np.pi*w1)*np.pi**2+3*FREQUENCY**2*torch.sin(FREQUENCY*np.pi*w3)*torch.sin(FREQUENCY*np.pi*w2)*torch.sin(FREQUENCY*np.pi*w1)*np.pi**2
                    loss_i = torch.norm(Fu)**2/batchsize


                    #boundary loss
                    w1_b1 = 0*torch.ones(batchsize_b,1, device=device, requires_grad=True) # (batch, 1)
                    w1_b2 = 1*torch.ones(batchsize_b,1, device=device, requires_grad=True)
                    w2_1 = torch.rand(batchsize_b, 1, device=device, requires_grad=True)
                    w3_1 = torch.rand(batchsize_b, 1, device=device, requires_grad=True)
                    
                    z1=torch.cat([w1_b1,w2_1,w3_1], dim=-1).to(device)
                    z2=torch.cat([w1_b2,w2_1,w3_1], dim=-1).to(device)

                    U_1 = model(z1)
                    U_2 = model(z2)

                    w2_b1 = 0*torch.ones(batchsize_b,1, device=device, requires_grad=True) # (batch, 1)
                    w2_b2 = 1*torch.ones(batchsize_b, 1,device=device, requires_grad=True)
                    w1_2 = torch.rand(batchsize_b, 1, device=device, requires_grad=True)
                    w3_2 = torch.rand(batchsize_b, 1, device=device, requires_grad=True)
                    
                    z3=torch.cat([w1_2,w2_b1,w3_2], dim=-1).to(device)
                    z4=torch.cat([w1_2,w2_b2,w3_2], dim=-1).to(device)

                    U_3 = model(z3)
                    U_4 = model(z4)

                    w3_b1 = 0*torch.ones(batchsize_b,1, device=device, requires_grad=True) # (batch, 1)
                    w3_b2 = 1*torch.ones(batchsize_b, 1,device=device, requires_grad=True)
                    w1_3 = torch.rand(batchsize_b, 1, device=device, requires_grad=True)
                    w2_3 = torch.rand(batchsize_b, 1, device=device, requires_grad=True)
                    
                    z5=torch.cat([w1_3,w2_3,w3_b1], dim=-1).to(device)
                    z6=torch.cat([w1_3,w2_3,w3_b2], dim=-1).to(device)

                    U_5 = model(z5)
                    U_6 = model(z6)

                    loss_b = torch.norm(U_1)**2/batchsize_b + torch.norm(U_2)**2/batchsize_b+torch.norm(U_3)**2/batchsize_b + torch.norm(U_4)**2/batchsize_b+torch.norm(U_5)**2/batchsize_b + torch.norm(U_6)**2/batchsize_b
                    loss = weight*loss_i +1*loss_b 



                    # true loss
                    Utruth = torch.sin(np.pi*w3)*torch.sin(np.pi*w2)*torch.sin(np.pi*w1)+torch.sin(FREQUENCY*np.pi*w3)*torch.sin(FREQUENCY*np.pi*w2)*torch.sin(FREQUENCY*np.pi*w1)
                    loss_truth = torch.norm(Utruth.reshape(-1, 1) - U)**2/batchsize

                    U1_t = grad(Utruth.sum(), w1, create_graph=True)[0]
                    U2_t = grad(Utruth.sum(), w2, create_graph=True)[0]
                    U3_t = grad(Utruth.sum(), w3, create_graph=True)[0]
                    loss_truth_h1 = torch.norm(U1_t - U1)**2/batchsize + torch.norm(U2_t - U2)**2/batchsize+ torch.norm(U3_t - U3)**2/batchsize


                    true_losses_l2.append(loss_truth.item())
                    true_losses_h1.append(loss_truth_h1.item())
                    if ep % 1 == 0:
                        print("Epoch is {}, interior loss is {}, boundary loss is {},  overall loss is {}, l2 true loss is {}, h1 true loss is {}".format(ep, loss_i.item(),loss_b.item(), loss.item(),loss_truth.item(),loss_truth_h1.item()))
                       
        if DIM ==2:

            w1 = torch.rand(batchsize, 1, requires_grad=True, device=device)
            w2 = torch.rand(batchsize, 1, requires_grad=True, device=device)
            y=torch.cat([w1,w2], dim=-1).to(device)
            for grid in grids:
                print('grid =',grid)

                def closure():
                    optimizer.zero_grad()
                    U = model(y)
                    U1 = grad(U.sum(), w1, create_graph=True)[0]
                    U2 = grad(U.sum(), w2, create_graph=True)[0]
                    U11 = grad(U1.sum(), w1, create_graph=True)[0]
                    U22 = grad(U2.sum(), w2, create_graph=True)[0]
                    #interior and smoothness losses
                    Fu = U22+U11 +2*torch.sin(np.pi*w2)*torch.sin(np.pi*w1)*np.pi**2+2*FREQUENCY**2*torch.sin(FREQUENCY*np.pi*w2)*torch.sin(FREQUENCY*np.pi*w1)*np.pi**2
                    loss_i = torch.norm(Fu)**2/batchsize


                    #boundary loss
                    w1_b1 = 0*torch.ones(batchsize_b,1, device=device, requires_grad=True) # (batch, 1)
                    w1_b2 = 1*torch.ones(batchsize_b,1, device=device, requires_grad=True)
                    w2_0 = torch.rand(batchsize_b, 1, device=device, requires_grad=True)

                    z1=torch.cat([w1_b1,w2_0], dim=-1).to(device)
                    z2=torch.cat([w1_b2,w2_0], dim=-1).to(device)
                    U_1 = model(z1)
                    U_2 = model(z2)


                    w2_b1 = 0*torch.ones(batchsize_b,1, device=device, requires_grad=True) # (batch, 1)
                    w2_b2 = 1*torch.ones(batchsize_b,1, device=device, requires_grad=True)
                    w1_0 = torch.rand(batchsize_b, 1, device=device, requires_grad=True)

                    z3=torch.cat([w1_0,w2_b1], dim=-1).to(device)
                    z4=torch.cat([w1_0,w2_b2], dim=-1).to(device)
                    U_3 = model(z3)
                    U_4 = model(z4)
                    loss_b = torch.norm(U_1)**2/batchsize_b + torch.norm(U_2)**2/batchsize_b+torch.norm(U_3)**2/batchsize_b + torch.norm(U_4)**2/batchsize_b
                    loss = weight*loss_i +1*loss_b 
                #back propogation of losses
                    loss.backward()
                    return loss

                for ep in range(step):
                    # 2d grid points
                    optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

                    if ep==0 and model_choice == "KAN":
                        model2 = KAN(width=np.array([DIM] + [WIDTH]*(DEPTH-1) + [1]).tolist(), grid=grid, k=3, symbolic_enabled=False, auto_save=True).to(device)
                        if grid != grids[0]:
                            model2.initialize_from_another_model(model, y)
                            model = model2
                            model.to(device)
                            print(model.act_fun[0].grid.device)
                        optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)


                    if ep % update_grid_freq == 0 and ep <= stop_update_grid_step and model_choice == "KAN":
                        model.update_grid_from_samples(y)

                    optimizer.step(closure)

                    # evaluation
                    
                    U = model(y)
                    U1 = grad(U.sum(), w1, create_graph=True)[0]
                    U2 = grad(U.sum(), w2, create_graph=True)[0]
                    U11 = grad(U1.sum(), w1, create_graph=True)[0]
                    U22 = grad(U2.sum(), w2, create_graph=True)[0]
                    #interior and smoothness losses
                    Fu = U22+U11 +2*torch.sin(np.pi*w2)*torch.sin(np.pi*w1)*np.pi**2+2*FREQUENCY**2*torch.sin(FREQUENCY*np.pi*w2)*torch.sin(FREQUENCY*np.pi*w1)*np.pi**2
                    loss_i = torch.norm(Fu)**2/batchsize


                    #boundary loss
                    w1_b1 = 0*torch.ones(batchsize_b,1, device=device, requires_grad=True) # (batch, 1)
                    w1_b2 = 1*torch.ones(batchsize_b,1, device=device, requires_grad=True)
                    w2_0 = torch.rand(batchsize_b, 1, device=device, requires_grad=True)

                    z1=torch.cat([w1_b1,w2_0], dim=-1).to(device)
                    z2=torch.cat([w1_b2,w2_0], dim=-1).to(device)
                    U_1 = model(z1)
                    U_2 = model(z2)


                    w2_b1 = 0*torch.ones(batchsize_b,1, device=device, requires_grad=True) # (batch, 1)
                    w2_b2 = 1*torch.ones(batchsize_b,1, device=device, requires_grad=True)
                    w1_0 = torch.rand(batchsize_b, 1, device=device, requires_grad=True)

                    z3=torch.cat([w1_0,w2_b1], dim=-1).to(device)
                    z4=torch.cat([w1_0,w2_b2], dim=-1).to(device)
                    U_3 = model(z3)
                    U_4 = model(z4)
                    loss_b = torch.norm(U_1)**2/batchsize_b + torch.norm(U_2)**2/batchsize_b+torch.norm(U_3)**2/batchsize_b + torch.norm(U_4)**2/batchsize_b
                    loss = weight*loss_i +1*loss_b 

                    # true loss
                    Utruth = torch.sin(np.pi*w2)*torch.sin(np.pi*w1)+torch.sin(FREQUENCY*np.pi*w1)*torch.sin(FREQUENCY*np.pi*w2)
                    loss_truth = torch.norm(Utruth.reshape(-1, 1) - U)**2/batchsize

                    U1_t = grad(Utruth.sum(), w1, create_graph=True)[0]
                    U2_t = grad(Utruth.sum(), w2, create_graph=True)[0]
                    loss_truth_h1 = torch.norm(U1_t - U1)**2/batchsize + torch.norm(U2_t - U2)**2/batchsize

                    true_losses_l2.append(loss_truth.item())
                    true_losses_h1.append(loss_truth_h1.item())
                    if ep % 1 == 0:
                        print("Epoch is {}, interior loss is {}, boundary loss is {},  overall loss is {}, l2 true loss is {}, h1 true loss is {}".format(ep, loss_i.item(),loss_b.item(), loss.item(),loss_truth.item(),loss_truth_h1.item()))

                    # save loss curves
        np.savetxt(f'l2loss_dim_{DIM}_frequency_{FREQUENCY}_depth_{DEPTH}_model_{model_choice}.txt', np.array(true_losses_l2))
        np.savetxt(f'h1loss_dim_{DIM}_frequency_{FREQUENCY}_depth_{DEPTH}_model_{model_choice}.txt', np.array(true_losses_h1))

run()
