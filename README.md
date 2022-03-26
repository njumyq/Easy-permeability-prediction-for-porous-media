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
From feature_extraction directory, run:`python feature_extraction.py` 

Then you could achieve **porosity_2d.csv**, **pecific_perimeter.csv**, **euler_number.csv** and **euler_number_std.csv** of the test samples.

## Machine learning models
## LSTM model
<div align=center><img width="900" src="https://github.com/499683288/Easy-permeability-prediction-for-porous-media/blob/main/LSTM%20model.jpg"/></div>
