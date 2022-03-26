import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import pandas as pd
from sklearn.metrics import r2_score 
import warnings
warnings.filterwarnings("ignore")


class RNN(nn.Module):
    def __init__(self, h_RNN_layers=1, h_RNN=64, h_FC_dim=256, drop_p=0, num_classes=1):
        super(RNN,self).__init__() 
        self.RNN_input_size = 3                     
        self.h_RNN_layers = h_RNN_layers   
        self.h_RNN = h_RNN                          
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes
        
        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,   
            hidden_size=self.h_RNN,              
            num_layers=h_RNN_layers,       
            batch_first=True,                      
        )
        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)
       
    def forward(self,X):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(X, None)
        x = self.fc1(RNN_out[:, -1, :])
        x = torch.relu(x)
        x = self.fc2(x)
        return x

def train(log_interval, model, device, train_loader, optimizer, epoch):
    rnn = model
    rnn.train()
    scores= []
    losses = []
    N_count = 0   
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device).view(-1, )
        N_count += X.size(0)
        optimizer.zero_grad()   
        output = rnn(X)                     
        MSE = nn.MSELoss(reduction='mean')
        loss = torch.sqrt(MSE(output.squeeze(-1), y))    
        losses.append(loss.item())
        y_pred = output
        loss.backward()
        optimizer.step()
        step_score = r2_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f},Accuracy: {:.4f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(),step_score))      
    return losses,scores 

def validation(model, device, optimizer, test_loader):
    rnn= model
    rnn.eval()
    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device).view(-1, )
            output = rnn(X)
            MSE = nn.MSELoss(reduction='sum')      
            loss = MSE(output.squeeze(-1), y)
            test_loss += loss.item()                            
            y_pred = output
            all_y.extend(y)
            all_y_pred.extend(y_pred)
    test_loss /= len(test_loader.dataset)
    test_loss=torch.sqrt(torch.tensor(test_loss))
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = r2_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    if epoch == best_epoch+1 :    
        M = np.array(all_y.cpu().data.squeeze().numpy())
        N = np.array( all_y_pred.cpu().data.squeeze().numpy())
        np.savetxt("./all_test_y.txt",M)
        np.savetxt("./all_pred_y.txt",N)
    #notice: they are not the best result
 
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.4f}\n'.format(len(all_y), test_loss, test_score))
    return test_loss,test_score

RNN_hidden_layers = 4
RNN_hidden_nodes = 4096
RNN_FC_dim = 2048
k = 1    
batch_size = 32
epochs = 500      
learning_rate = 1e-4
log_interval = 32
dropout_p=0
use_cuda = torch.cuda.is_available()                  
device = torch.device("cuda" if use_cuda else "cpu")   
sample_number = 5000
slice_number = 256


path1=os.getcwd()+'/specific_perimeter.csv'
path2=os.getcwd()+'/porosity_2d.csv'
path3=os.getcwd()+'/euler_number_std.csv'
path4=os.getcwd()+'/permeability.csv'
f1=open(path1,encoding='utf-8')
f2=open(path2,encoding='utf-8')
f3=open(path3,encoding='utf-8')
f4=open(path4,encoding='utf-8')
X1=pd.read_csv(f1,low_memory=False,header=None)
X2=pd.read_csv(f2,low_memory=False,header=None)
X3=pd.read_csv(f3,low_memory=False,header=None)
X1,X2,X3=np.array(X1),np.array(X2),np.array(X3)
X1,X2,X3=torch.FloatTensor(X1).unsqueeze(-1).numpy(),torch.FloatTensor(X2).unsqueeze(-1).numpy(),torch.FloatTensor(X3).unsqueeze(-1).numpy()
X=np.concatenate((X1,X2,X3),axis=-1)

y=pd.read_csv(f4,low_memory=False,header=None)
y= y.values

use_cuda = torch.cuda.is_available()                   
device = torch.device("cuda" if use_cuda else "cpu") 

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.30, random_state=0)
train_x=torch.FloatTensor(train_x)
test_x=torch.FloatTensor(test_x)
train_y=torch.FloatTensor(train_y)
test_y=torch.FloatTensor(test_y)
train_dataset = data.TensorDataset(train_x, train_y)
test_dataset=data.TensorDataset(test_x,test_y)

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=False)
valid_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=False)

rnn = RNN(h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)
rnn_params = list(rnn.parameters())
optimizer = torch.optim.Adam(rnn_params, lr=learning_rate, weight_decay=0)
    
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

def adjust_learning_rate(optimizer, epoch):
    if epoch ==50:
        lr = learning_rate * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif epoch==150:
        lr = learning_rate * 0.05
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif epoch==250:
        lr = learning_rate * 0.025
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif epoch==350:
        lr = learning_rate * 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr        
    
best_score=0
best_epoch=0
save_path="./lstm_model.pth"

for epoch in range(epochs):
    adjust_learning_rate(optimizer,epoch)
    train_loss,train_scores = train(log_interval,  rnn, device, train_loader, optimizer, epoch)
    epoch_test_loss,epoch_test_score = validation(rnn, device, optimizer, valid_loader)
    if best_score < epoch_test_score:
        best_score=epoch_test_score
        best_epoch=epoch
        #torch.save(rnn.state_dict(),save_path)
    print("best_score is {:.4f}\n".format(best_score) )
    print("best_epoch is {:.4f}\n".format(best_epoch) )

    epoch_train_losses.append(train_loss)
    epoch_test_losses.append(epoch_test_loss)
    epoch_train_scores.append(train_scores)
    epoch_test_scores.append(epoch_test_score)
    
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)
    np.save('./epoch_training_losses.npy', A)
    np.save('./epoch_test_loss.npy', C)
    np.save('./epoch_training_scores.npy', B)
    np.save('./epoch_test_scores.npy', D)