import numpy as np
import torch.nn as nn
from kan import *
from PIL import Image
import time
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

pic_ids = [0]#[0,1,2]
grids = [[100,10,10,10,10,10]]#,[10,10,10,10,10],[100,100,100,100,100]]

for pic_id in pic_ids:
    for grid in grids:
        pic_id = 0
        if pic_id == 0:
            image = np.array(Image.open('./cameraman.jpg').convert('L'))
            #if grid == [100,10,10,10,10] or grid == [10,10,10,10,10]:
                #continue
        elif pic_id == 1:
            image = np.array(Image.open('./turbulence.png').convert('L'))
        else:
            image = np.array(Image.open('./starrynight.png').convert('L'))

        print(f'pic_id={pic_id}, grid={grid}')
            
        image = 2*(image/256 - 0.5)

        dimx, dimy = image.shape
        x_grid = np.linspace(-1,1,num=dimx)
        y_grid = np.linspace(-1,1,num=dimy)
        xx, yy = np.meshgrid(x_grid, y_grid)
        inputs = np.transpose(np.array([xx.reshape(-1,), yy.reshape(-1,)]))
        labels = image.reshape(-1,)
        num = labels.shape[0]


        dataset = {}
        dataset['train_input'] = torch.tensor(inputs, dtype=torch.float32, requires_grad=True).to(device)
        dataset['train_label'] = torch.tensor(labels[:,np.newaxis], dtype=torch.float32, requires_grad=True).to(device)

        dataset['test_input'] = torch.tensor(inputs, dtype=torch.float32, requires_grad=True).to(device)
        dataset['test_label'] = torch.tensor(labels[:,np.newaxis], dtype=torch.float32, requires_grad=True).to(device)

        def PSNR(original, compressed): 
            mse = np.mean((original - compressed) ** 2) 
            if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                          # Therefore PSNR have no importance. 
                return 100
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) 
            return psnr

        steps = 5000
        model = KAN(width=[2,256,128,128,128,128,1], grid=grid, device=device)
        model.speed()
        train_losses = []
        test_losses = []

        start_time = time.time()

        loss_fn = nn.MSELoss()
        grid_update_freq = 10
        stop_grid_update_step=1000
        batch=4096
        #optim = torch.optim.Adam(list(model.parameters()), lr=1e-3)
        optim = torch.optim.Adam(list(model.act_fun[-1].parameters())+list(model.act_fun[1].parameters())+list(model.act_fun[2].parameters())+list(model.act_fun[3].parameters())+list(model.act_fun[4].parameters()), lr=1e-3)
        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        for iter_num in range(steps):
            train_id = np.random.choice(dataset['train_input'].shape[0], batch, replace=False)
            x=dataset['train_input'][train_id]
            if iter_num % grid_update_freq == 0 and iter_num < stop_grid_update_step:
                model.update_grid_from_samples(x)
            optim.zero_grad()
            y = model(x)
            loss = loss_fn(y, dataset['train_label'][train_id])
            loss.backward()
            optim.step()
            #test_losses = loss_fn_eval(model(dataset['test_input']), dataset['test_label'])
            results['train_loss'].append(torch.sqrt(loss).cpu().detach().numpy())
            #results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
            


        train_losses += results['train_loss']
        #optim = torch.optim.Adam(list(model.parameters()), lr=1e-4)
        optim = torch.optim.Adam(list(model.act_fun[-1].parameters())+list(model.act_fun[1].parameters())+list(model.act_fun[2].parameters())+list(model.act_fun[3].parameters())+list(model.act_fun[4].parameters()), lr=1e-4)
        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        for iter_num in range(steps):
            train_id = np.random.choice(dataset['train_input'].shape[0], batch, replace=False)
            x=dataset['train_input'][train_id]
            if iter_num % grid_update_freq == 0 and iter_num < stop_grid_update_step:
                model.update_grid_from_samples(x)
            optim.zero_grad()
            y = model(x)
            loss = loss_fn(y, dataset['train_label'][train_id])
            loss.backward()
            optim.step()
            #test_losses = loss_fn_eval(model(dataset['test_input']), dataset['test_label'])
            results['train_loss'].append(torch.sqrt(loss).cpu().detach().numpy())
            #results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
        train_losses += results['train_loss']
        
        #optim = torch.optim.Adam(list(model.parameters()), lr=1e-5)
        optim = torch.optim.Adam(list(model.act_fun[-1].parameters())+list(model.act_fun[1].parameters())+list(model.act_fun[2].parameters())+list(model.act_fun[3].parameters())+list(model.act_fun[4].parameters()), lr=1e-5)
        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        for iter_num in range(steps):
            train_id = np.random.choice(dataset['train_input'].shape[0], batch, replace=False)
            x=dataset['train_input'][train_id]
            if iter_num % grid_update_freq == 0 and iter_num < stop_grid_update_step:
                model.update_grid_from_samples(x)
            optim.zero_grad()
            y = model(x)
            loss = loss_fn(y, dataset['train_label'][train_id])
            loss.backward()
            optim.step()
            #test_losses = loss_fn_eval(model(dataset['test_input']), dataset['test_label'])
            results['train_loss'].append(torch.sqrt(loss).cpu().detach().numpy())
            #results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
        train_losses += results['train_loss']
        #results = model.fit(dataset, opt='Adam', steps=steps, update_grid=False, batch=4096, lr=1e-5);
        #train_losses += results['train_loss']

        end_time = time.time()


        batch = 4096
        n_batch = inputs.shape[0]//batch + 1
        for i in range(n_batch):
            if i % 20 == 0:
                print(i)
            data_batch = torch.tensor(inputs[i*batch:(i+1)*batch], dtype=torch.float32).to(device)
            if i == 0:
                out = model(data_batch).cpu().detach()
            else:
                out = torch.cat([out, model(data_batch).cpu().detach()], dim=0)

        wall_time = end_time - start_time

        compressed = (out[:,0].reshape(dimx,dimy).detach().numpy() + 1)*128
        original = (image + 1) * 128
        psnr = PSNR(original, compressed)
        print('psnr=',psnr)
        plt.imshow(out[:,0].reshape(dimx,dimy).detach().numpy(), cmap='gray')
        plt.title('psnr=%.2f'%psnr, fontsize=15)
        plt.axis('off')

        #image
        plt.savefig(f'./results/kan_picture_{pic_id}_grid_{grid}.png', bbox_inches='tight')
        plt.clf()

        #wall time
        np.savetxt(f'./results/kan_walltime_{pic_id}_grid_{grid}', [wall_time])

        #losses
        np.savetxt(f'./results/kan_trainloss_{pic_id}_grid_{grid}', train_losses)
