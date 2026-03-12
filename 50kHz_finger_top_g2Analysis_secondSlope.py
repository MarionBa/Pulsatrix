import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import re
import pandas as pd
from pathlib import Path
import process
import torch
from scipy.signal import filtfilt, butter
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from matplotlib import cm
from scipy import signal#


### Load g2 function ###
folder = r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR_experiment\g2_maps\Finger_top\Short lag'
Files = list(Path(folder).glob('*.npy'))
print(Files)


# Parameters
final_fps = 100
fps = 50000
T = 20 # Total real time we want to see the signal
lag_i = 0.00006 # The lag time we want to consider
lag_f = 0.004

# Time arrays
timeLag = np.arange(lag_i, lag_f, 1/fps)

# Analysis
n = 0
Files = [Files[2], Files[5]] # Selecting only the 50kHz data
for file in Files:
    g2_map = np.load(file)

    ### Fitting ###
    LinCoef = []
    TimeFit = 20
    for g2 in g2_map:
        # plt.figure()
        # plt.semilogx(np.arange(1/fps, 0.1+1/fps, 1/fps),g2)
        # plt.xlim([0.00006, 0.004])
        #plt.show()

        temp = np.log(g2[int(lag_i*fps):int(lag_f*fps)])
        P = np.polyfit(np.arange(lag_i, lag_f, 1/fps), temp, 1)
        # Figure check
        # plt.figure()
        # plt.plot(timeLag, temp)
        # plt.plot(timeLag, timeLag*P[0] + P[1])
        # plt.show()
        LinCoef.append(P[0])

    # Plot coefs
    plt.figure()
    Sav_filt = savgol_filter(LinCoef, window_length=5, polyorder=1)
    plt.plot(np.arange(0, 20, 1/final_fps), Sav_filt)
    plt.xlabel('Time [s]')
    plt.ylabel('Linear coefficient')
    plt.legend(['Inverted signal'])
    plt.xlim([10.7, 11.9])

    plt.show()