from VSViG import *
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch, json
import torch.nn as nn
import numpy as np

class vsvig_dataset(Dataset):
    def __init__(self, data_folder=None, label_file=None, transform=None):
        super().__init__()
        self._folder = data_folder
        self._transform = transform
        with open(label_file, 'rb') as f:
            self._labels = json.load(f)
            

    def __getitem__(self,idx):
        target = float(self._labels[idx][1])
        data_idx = self._labels[idx][0]
        data = torch.load(PATH_TO_DATA) # Inputs: Batches, Frames, Points, Channles, Height, Width (B,30,15,3,32,32)
        kpts = torch.load(PATH_TO_KPTS) # (B, 15, 2), where 15 is number of kpts, and 2 means coordinates (x, y) of each kpt
        data = data.squeeze(0)
        raw_order = list(np.arange(18)) # raw 18 keypoint COCO template
        new_order = [0,-3,-4] + list(np.arange(12)+2) + [1,-1,-2] # reorder them
        kpts[:,raw_order,:] = kpts[:,new_order,:]
        if self._transform: # 30,15,3,32,32
            data = data.view(30*15*3,32,32)
            data = self._transform(data)
            data = data.view(30,15,3,32,32)
            
        sample = {
            'data': data,
            'kpts': kpts[:,:15,:]
            }
        return sample, target
    
    def __len__(self):
        return len(self._labels)

def train():
    dy_point_order = torch.load('PATH_TO_DYNAMIC_PARTITIONS')
    data_path = PATH_TO_DATA # Inputs: Batches, Frames, Points, Channles, Height, Width (B,30,15,3,32,32)
    label_path = PATH_TO_LABEL # Outputs/Labels: Batches, 1 (0 to 1 probabilities/likelihoods) (B,1) 
    models = ['base', 'light']
    
    for m in models:
        dataset_train = vsvig_dataset(data_folder=data_path, label_file=label_path, transform=None)
        dataset_val = vsvig_dataset(data_folder=data_path, label_file=label_path, transform=None)
        train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True,num_workers=4)
        val_loader = DataLoader(dataset_val, batch_size=32, shuffle=True,num_workers=4)
        
        # criterion = nn.BCEWithLogitsLoss()
        MSE = nn.MSELoss()
        epochs = 200
        min_valid_loss = np.inf
        if m == 'Base':
            model = VSViG_base()
        elif m == 'Light':
            model = VSViG_light()
        if torch.cuda.is_available():
            model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        train_loss_stack = []
        for e in range(epochs):
            train_loss = 0.0
            
            model.train()
            optimizer.zero_grad()
            print(f'===================================\n Running Epoch: {e+1} \n===================================')

            for sample, labels in train_loader:
                data = sample['data']
                kpts = sample['kpts']

                if torch.cuda.is_available():
                    data, labels, kpts = data.cuda(), labels.cuda(), kpts.cuda()
                outputs = model(data,kpts)
                # print(outputs)
                loss = MSE(outputs.float(),labels.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_loss_stack.append(loss.item())
            print(f'Training Loss: {train_loss:.3f}')

            if (e+1)%5 == 0:
                valid_loss = 0.0
                RMSE_loss = 0.0
                _iter = 0
                model.eval()

                for sample, labels in val_loader:
                    data = sample['data']
                    kpts = sample['kpts']
                    if torch.cuda.is_available():
                        data, labels, kpts = data.cuda(), labels.cuda(), kpts.cuda()
                    outputs = model(data,kpts)
                    loss = MSE(outputs,labels)
                    valid_loss += loss.item()
                    RMSE_loss += torch.sqrt(MSE(outputs,labels)).item()*100
                    _iter += 1
                print(f' +++++++++++++++++++++++++++++++++++\n Val Loss: {valid_loss:.3f} \t Val RMSE: {RMSE_loss/_iter:.3f} \n +++++++++++++++++++++++++++++++++++')

                if min_valid_loss > valid_loss:
                    print(f'save the model \n +++++++++++++++++++++++++++++++++++')
                    min_valid_loss = valid_loss
                    save_model_path = PATH_TO_MODEL
                    torch.save(model.state_dict(), save_model_path)
            scheduler.step()
                    
if __name__ == '__main__':
    train()
