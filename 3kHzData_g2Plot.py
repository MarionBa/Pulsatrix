"""
Created on Thu Jul 20 13:51:15 2023

@author: barbea43

Here we compute the autocorrelation function for intensity based sensors
see:
    https://opg.optica.org/boe/fulltext.cfm?uri=boe-8-11-4855&id=375024

    The data is extracted from .py array files and csv files
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, correlate,correlation_lags
from scipy.signal import savgol_filter
from scipy.signal import resample

fname = 'release'
folder = r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR'
g2_map = np.load(folder + '\\g2_map' + fname + '.npy')
xi = 0

DF = pd.read_csv(folder + '\\SPG_PPG_' + fname + '.csv')
PPG = DF.PPG.values[xi:]
SPG = DF.SPG.values[xi:]

# Calculation correlation time
fps = 3000
n = 300
dt = 1/ fps
tau_c_80 = []
tau_c_50 = []
tau_c_10 = []
for i in range(xi, len(g2_map)-n):

    g2_smooth = savgol_filter(g2_map[i], 30, 3)
    # Check plots
    # plt.figure()
    # plt.plot(g2_map[i])
    # plt.plot(g2_smooth)
    # plt.show()

    ### Correlation time calculation ###
    # upsample minority
    g2_smooth_upsampled = resample(g2_smooth, num=10*len(g2_smooth))
    g2_final = g2_smooth #savgol_filter(g2_smooth_upsampled, 30, 3)
    lag = np.arange(0, n / fps, 1 / fps)
    lag_upsampled = np.arange(0, n / fps, 1 / (fps*10))
    final = 1000
    # plt.figure()
    # plt.plot(lag[:int(final/10)], g2_smooth[:int(final/10)])
    # plt.plot(lag_upsampled[:final], g2_smooth_upsampled[:final])
    # plt.plot(lag_upsampled[:final], g2_final[:final])
    # plt.show()


    # 80%
    temp = np.where(g2_final <= g2_final[0] - (g2_final[0] - g2_final[final]) * 0.8) # len(g2_smooth_upsampled) - 1
    temp = temp[0]
    tau_c_80.append(temp[0] * dt)

    # 50%
    temp = np.where(g2_final <= g2_final[0] - (g2_final[0] - g2_final[final]) * 0.5)
    temp = temp[0]
    tau_c_50.append(temp[0] * dt)

    # 10%
    temp = np.where(g2_final <= g2_final[0] - (g2_final[0] - g2_final[final]) * 0.1) #len(g2_smooth_upsampled) - 1
    temp = temp[0]
    tau_c_10.append(temp[0] * dt)


### Figures ###

# G2 plots
plt.figure()
g2length = 300
n = 6000 - (g2length+1)
lag = np.arange(0,  g2length/fps, 1/fps) # Time array
plt.plot(lag, g2_map[int(0.003*3000)], lag, g2_map[int(0.013*3000)]) #, lag, g2_map[3300])
#plt.plot(lag, g2_map[450]) #, lag, g2_map[2400])
#plt.plot(lag, (g2_map[xi:, 150]-min(g2_map[xi:, 150]))/(max(g2_map[xi:, 150])-min(g2_map[xi:, 150]))+0.5)
plt.ylabel(r'g2($\tau$)')
plt.ylim(1.175, 1.188)
#plt.xlim(0, 0.135)
plt.xlabel(r'$\tau$ [s]')
#plt.show()

# g2 map
# plt.figure()
time = np.arange(xi/fps,  (n-1)/fps, 50/fps)
# plt.pcolormesh(time, lag[50:], g2_map[xi:, 50:]) #np.rot90(np.rot90(np.rot90(g2_map[xi:,50:]))))
# plt.ylabel(r'$\tau$ [s]')
# plt.xlabel('Time [s]')

# SPG/PPG plots
# plt.figure()
NormSPG = (SPG-min(SPG))/(max(SPG)-min(SPG))
NormPPG = (PPG-min(PPG))/(max(PPG)-min(PPG))
# plt.plot(NormPPG, 'r')
# plt.plot(NormSPG, 'b')
# plt.legend(['PPG', 'SPG'])
# plt.xlabel('Time [s]')



# Correlation time
# plt.figure()
# plt.plot(savgol_filter(tau_c_80, 50, 3))
# plt.plot(savgol_filter(tau_c_50, 50, 3))
# #plt.show()

plt.figure()
g2map = np.rot90(np.rot90(np.rot90(g2_map[::-1])))
plt.pcolormesh(time, lag[10:], g2map[10:])
plt.ylabel(r'$\tau$ [s]')
plt.xlabel('Time [s]')


plt.figure()
g2map = np.rot90(np.rot90(np.rot90(g2_map)))
plt.pcolormesh(time, lag[10:], g2map[10:])
plt.ylabel(r'$\tau$ [s]')
plt.xlabel('Time [s]')
#plt.show()

# G2 plots lateral
plt.figure()
mean1 = []
for i in np.arange(5, 50): #(int(0.0003*3000), int(0.003*3000)):
    mean1.append(g2_map[:, i]) #/(np.mean(g2_map[:, i])))
mean2 = []
for i in np.arange(100, 150):
    mean2.append(g2_map[:, i]) #/(np.mean(g2_map[:, i])))

mean1 = savgol_filter(mean1, 101, 3)
mean2 = savgol_filter(mean2, 101, 3)

plt.plot(time, np.mean(mean1, 0), time, np.mean(mean2, 0))
plt.ylabel(r'g2($\tau$)')
#plt.ylim(1.175, 1.188)
plt.xlabel(r'$Time$ [s]')



### Fancy big figure ###
fig, ax = plt.subplots(3, 1, layout='constrained', figsize=(8, 6))
ax[0].pcolormesh(time, lag, g2map)
ax[0].set_ylabel(r'$\tau$ [s]')
ax[0].xaxis.set_ticklabels([])

ax[1].plot(tau_c_10)
ax[1].plot(tau_c_50) #, 50, 3))
ax[1].plot(tau_c_80)
ax[1].legend([r'10% decay g2($\tau$)', r'50% decay g2($\tau$)', r'80% decay g2($\tau$)'])
ax[1].set_xlim([0, len(tau_c_50)])
ax[1].xaxis.set_ticklabels([])

ax[2].plot(NormPPG,'r')
ax[2].plot(NormSPG,'b')
ax[2].legend(['SPG', 'PPG'])
ax[2].set_xlabel('Time [s]')
ax[2].set_xlim([0, len(SPG)])
ax[2].xaxis.set_ticklabels([])
plt.show()





