import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.mlab as mlab
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from matplotlib.pyplot import MultipleLocator
from scipy.stats import gaussian_kde

epochs=500
def model_loss_score(epochs):
    A_=np.load("./epoch_training_losses.npy")
    B_=np.load("./epoch_training_scores.npy")
    C_=np.load("./epoch_test_loss.npy")
    D_=np.load("./epoch_test_scores.npy")

    fig1 = plt.figure(figsize=(10, 4))  
    plt.plot(np.arange(1, epochs + 1), A_[:, -1],"b")  
    plt.plot(np.arange(1, epochs + 1), C_,"r")         
    plt.title("model loss",fontsize=16)
    plt.xlabel('epochs',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.ylim(0,0.5)
    plt.grid('minor')
    plt.legend(['train', 'test'], loc="upper right",fontsize=14)
    plt.tick_params(which='minor',direction='in')
    plt.tick_params(labelsize=14)  
    title1 = "./model_loss.png"
    plt.savefig(title1,dpi=900,bbox_inches="tight")

    fig2 = plt.figure(figsize=(10, 4))
    plt.plot(np.arange(1, epochs + 1), B_[:, -1],"b")  
    plt.plot(np.arange(1, epochs + 1), D_,"r")      
    plt.title("model score",fontsize=16)
    plt.grid('minor')
    plt.xlabel('epochs',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.ylim(0.5,1)
    plt.legend(['train', 'test'], loc="lower right",fontsize=14)
    plt.tick_params(which='minor',direction='in')
    plt.tick_params(labelsize=14) 
    title2 = "./model_score.png"
    plt.savefig(title2, dpi=500,bbox_inches="tight")

def best_result():
    plt.figure(figsize=(6,4))
    A=pd.read_table("./all_test_y.txt",low_memory=False,header=None)
    A=np.array(A)
    A=A.squeeze(-1)
    B=pd.read_table("./all_pred_y.txt",low_memory=False,header=None)
    B=np.array(B)
    B=B.squeeze(-1)
    scores=r2_score(B,A)
    RMSE=pow(mean_squared_error(abs(B),abs(A)),0.5)
    plt.minorticks_on()
    plt.tick_params(which='minor',direction='in')
    xy = np.vstack([A,B])
    z = gaussian_kde(xy)(xy)
    plt.scatter(A, B, marker='o',c=z, edgecolors=None,s=30, cmap="jet")

    plt.ylabel("Predicted Permeability",fontsize=16)
    plt.xlabel("True Permeability",fontsize=16)

    plt.title("Best Result",fontsize=16)
    colorbar=plt.colorbar()
    colorbar.ax.tick_params(labelsize=12)
    title = "./Best Result.png"
    plt.savefig(title, dpi=500,bbox_inches="tight")

model_loss_score(epochs)
best_result()