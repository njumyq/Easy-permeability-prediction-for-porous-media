import os
import numpy as np
import pandas as pd
from skimage.measure import euler_number, label

def extract_porosity(sample_number, slice_number):
    for file in range(1,sample_number+1):
        file_path='D:/python/code/feature_extraction/test_sample/' + str(file)
        list_=[]
        for j in range(1,slice_number+1):
            path=file_path+"/"+ str(j)
            pic=np.loadtxt(path +".csv",delimiter=",",dtype=int)
            count0,count1=0,0
            for m in range(0,pic.shape[0]):
                for n in range(0,pic.shape[1]):
                    if pic[m,n]==0:
                        count0+=1
                    else: 
                        count1+=1
            porosity=count0/(count0+count1)
            list_.append(porosity)
        porosity_2d=pd.DataFrame([list_])
        porosity_2d.to_csv('./porosity_2d.csv', mode='a', header=None, index=None)

def extract_specific_perimeter(sample_number, slice_number):
    for file in range(1,sample_number+1):
        file_path='D:/python/code/feature_extraction/test_sample/' + str(file)
        list_=[]
        for j in range(1,slice_number+1):
            path=file_path+"/"+ str(j)
            pic=np.loadtxt(path +".csv",delimiter=",",dtype=int)
            count0,count1=0,0
            count=0
            for m in range(0,pic.shape[0]):
                for n in range(0,pic.shape[1]):
                    if pic[m,n]==0:
                        count0+=1
                    else: 
                        count1+=1
            for i in range(1,int(pic.shape[0])-1):
                for j in range(1,int(pic.shape[1])-1):
                    if pic[i-1,j]==1 and pic[i+1,j]==1 and pic[i,j+1]==1 and pic[i,j-1]==1:   
                        count+=1
            specific_surface = (count1-count)/count0
            list_.append(specific_surface)

        specific_surface_2d=pd.DataFrame([list_])
        specific_surface_2d.to_csv('./specific_perimeter.csv', mode='a', header=None, index=None)

def extract_euler_number(sample_number, slice_number):
    for file in range(1,sample_number+1):
        file_path='D:/python/code/feature_extraction/test_sample/' + str(file)
        list_euler_number=[]
        for j in range(1,slice_number+1):
            path=file_path+"/"+ str(j)
            pic=pd.read_csv(path +".csv",low_memory=False,header=None)
            pic=pic.replace(0,2)
            pic=pic.replace(1,0)
            pic=pic.replace(2,1)
            pic = np.pad(pic, 1, mode='constant')
            euler = euler_number(pic, connectivity=1)
            list_euler_number.append(euler)
        euler_number_=pd.DataFrame([list_euler_number])
        euler_number_.to_csv('./euler_number.csv', mode='a', header=None, index=None)

    path='./euler_number.csv'
    f=open(path,encoding='utf-8')
    en=pd.read_csv(f,low_memory=False,header=None,index_col=False)
    en_=np.array(en)
    enmax,enmin=en_.max(axis=1).reshape(-1,1), en_.min(axis=1).reshape(-1,1)
    en_norm=(en_-enmin)/(enmax-enmin)
    en_norm=pd.DataFrame(en_norm)
    en_norm.to_csv('./euler_number_std.csv', header=None,index=None)

sample_number = 10
slice_number = 256

extract_porosity(sample_number, slice_number)
extract_specific_perimeter(sample_number, slice_number)
extract_euler_number(sample_number, slice_number)