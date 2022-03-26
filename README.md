# Easy-permeability-prediction-for-porous-media
Feature extraction of porous media; permeability prediction; machine learning; LSTM
## Description

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
From **machine learning(case1)** directory,open and run **ml for porosity sequences.ipynb**, **ml for specific perimeter sequences.ipynb** and **ml for euler number sequences.ipynb**, respectively, then you can get the permeability perdiction results with four machine learning models.

The parameter search of the model can be referred to as follows：
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
<div align=center><img width="900" src="https://github.com/499683288/Easy-permeability-prediction-for-porous-media/blob/main/LSTM%20model.jpg"/></div>
