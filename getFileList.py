import os
import numpy as np
import h5py
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd


path = "../dataset/" #文件夹目录
dir = os.listdir(path) #得到文件夹下的所有文件名称
f =  open("files.txt", "w")
for file in dir: #遍历文件夹
     if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
        f.write(path+"/"+file+"\n")


