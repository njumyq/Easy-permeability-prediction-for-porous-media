# Easy-permeability-prediction-for-porous-media
Feature extraction of porous media; permeability prediction; machine learning; long short-term memory neural network(LSTM)
## Description
This repository aims to provide a convenient method to predict the permeability of porous media with machine learning. 
## Installation
* numpy
* pandas
* matplotlib
* sklearn
* scipy
* scikit-image
* PyTorch
## Data preparation
<div align=center><img width="400" src="https://github.com/499683288/Easy-permeability-prediction-for-porous-media/blob/main/sample1.jpg"/></div>
<div align=center><img width="300" src="https://github.com/499683288/Easy-permeability-prediction-for-porous-media/blob/main/sample2.jpg"/></div>  

You should pre-express the 3D image of the porous medium as a folder composed of multiple slices (csv file format or image format). The name of each csv file is the slice number, and the folder name is the sample number.
## Feature extraction
<div align=center><img width="900" src="https://github.com/499683288/Easy-permeability-prediction-for-porous-media/blob/main/feature%20extraction.jpg"/></div>


From **feature_extraction** directory, run`python feature_extraction.py`.
Then you could achieve **porosity_2d.csv**, **pecific_perimeter.csv**, **euler_number.csv** and **euler_number_std.csv** of the test samples.

## Machine learning models
From **machine learning(case1)** directory,open and run **ml for porosity sequences.ipynb**, **ml for specific perimeter sequences.ipynb** and **ml for euler number sequences.ipynb** respectively by Juputer notebook, then you can get the permeability perdiction results with four machine learning models. **Visualization.ipynb** provides the visualization of predicted results. 

The parameter search of the models (without linear regression and LSTM) can be referred to as follows：
```
from sklearn.model_selection import GridSearchCV
param_grid1={"n_neighbors":range(1,20), "weights":["uniform","distance"],"algorithm":["kd_tree","auto"]}
grid_search1=GridSearchCV(KNeighborsRegressor(),param_grid1,cv=5)
grid_search1.fit(train_x,train_y)
print(grid_search1.best_params_)

param_grid2={"n_estimators":[50,100,500,1000], "oob_score":["True","False"],"max_features":["auto","log2","sqrt"],"max_depth":[5,10,50,100],
             "max_leaf_nodes":[5,10,30,50]}
grid_search2=GridSearchCV(RandomForestRegressor(),param_grid2,cv=5)
grid_search2.fit(train_x,train_y)
print(grid_search2.best_params_)

param_grid3={"C":[0.00001,0.0001,0.001,0.01,1,10],"kernel":["linear","rbf","sigmoid"]}
grid_search3=GridSearchCV(SVR(),param_grid3,cv=5)
grid_search3.fit(train_x,train_y)
print(grid_search3.best_params_)
```

## LSTM model
You should install Pytorch 1.7 and above. Then set the hyperparameter, such as
```
RNN_hidden_layers = 4
RNN_hidden_nodes = 4096
RNN_FC_dim = 2048
k = 1    
batch_size = 32
epochs = 500 
```
and run `python LSTM.py`. To get the relevant visualization, run `python Visualization.py`.

<div align=center><img width="900" src="https://github.com/499683288/Easy-permeability-prediction-for-porous-media/blob/main/LSTM%20model.jpg"/></div>

```
class RNN(nn.Module):
    def __init__(self, h_RNN_layers=1, h_RNN=64, h_FC_dim=256, drop_p=0, num_classes=1):
        super(RNN,self).__init__() 
        self.RNN_input_size = 3         # LSTM inputs three features in a time step           
        self.h_RNN_layers = h_RNN_layers  # Hidden layers of LSTM
        self.h_RNN = h_RNN              # The number of hidden layer neurons           
        self.h_FC_dim = h_FC_dim        # The number of neurons in the middle layer 
        self.drop_p = drop_p            # The proportion of neurons to be discarded 
        self.num_classes = num_classes  # For regression problem, num_class = 1
     
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
``` 

You can adjust the network structure and hyperparameters according to the output results. It should also be noted that the loss gradient of the LSTM model is unstable, and the loss function curve and the score curve are difficult to converge. You can conduct multiple model tests with the same parameters at the same time to obtain the best prediction results.
