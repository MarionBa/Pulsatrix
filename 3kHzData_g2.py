"""
Created on Thu Jul 20 13:51:15 2023

@author: barbea43

Here we compute the g2 autocorrelation function for intensity based sensors
see:
    https://opg.optica.org/boe/fulltext.cfm?uri=boe-8-11-4855&id=375024

    The 3kHz data is extracted from matlab files.
    Note: this code has been made to process the 3kHz data and processing with this code takes
    ~6h for 3seconds of video
"""

### Import packages ###
import numpy as np
from OwlProcess.processSPG import *
import pandas as pd
import h5py
import process
import NCC_functions
import process
from PIL import Image
import os
import NCC_functions
import process
import pandas as pd
import pims
import h5py
import glob


### Import files ###
# fname = 'release'
# path = r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR'
# video = h5py.File(path + '\\Subject2_' + fname + '_6Seconds_PythonFormat.mat', 'r')
# Tensor = np.array(video['data'])

dirname = r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR_experiment\3kHz\Sub 1 Finger_top'
file = glob.glob(f'{dirname}/*.seq')
final = pims.open(file[0])
length, x_pixels, y_pixels,  = final.shape
# Reshaping the tensor #
Tensor = []
for frame in final:
    Tensor.append(frame)
    # plt.figure()
    # plt.imshow(frame)
    # plt.show()


### Parameters ###
g2length = 300
n = 6000 - (g2length+1)

### Array initialization ###
g2_map = [] # g2 tensor map
SPG = []
PPG = []
NCC_map = []

### Get the g2 function map aka time as a function of tau as well as SPG and PPG ###
#Tensor = Tensor[1000:6000] # For speckle video time selection
for j in np.arange(0, n-1, 1):
    print((j/(n-1))*100, '% finished') # Processing % left
    temp = np.array(Tensor[j])
    T = temp
    # T = temp[328:404, 125:263] # For ROI selection
    # Image check
    # plt.figure()
    # plt.imshow(T)
    # plt.show()
    print('yo')
    # SPG and PPG calculation
    SPG.append(process.getSPG(T, method='sigma'))
    PPG.append(T[np.isfinite(T)].mean())

    # g2 function calculation
    g2 = []
    for i in np.arange(0, g2length):
        Num = np.mean(np.array(Tensor[j]) * np.array(Tensor[j+i]))
        Den = np.square(np.mean(np.array(Tensor[j]))) #* np.mean(np.array(Tensor[j+i]))
        g2.append(Num / Den)
        plt.figure()
        plt.plot(g2)
        plt.show()
    g2_map.append(g2/np.mean(g2))




### g2 tensor map saving as numpy array and SPG and PPG saving in csv format ###
#np.save(path + '\\g2_map' + fname + '_Otherg2Equation.npy', g2_map)
# df = pd.DataFrame({'SPG': SPG, 'PPG': PPG})
# df.to_csv(path + '\\SPG_PPG_' + fname + '.csv')





