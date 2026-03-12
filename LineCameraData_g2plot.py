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

Position = r'Neck'
Brownian = False # Draw line on your g2 function plot at time when brownian motion is supposed to take place
AllVelocities = True

### Load SPG and PPG ###
folder=r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR_experiment\SPG\\' + Position
#folder=r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR_experiment\SPG\Neck\3kHz data'
Files = list(Path(folder).glob('*.csv'))
print(Files)


SPG_array = []
PPG_array = []
SubjectList = []
fpsList = []
for file in Files:
    DF = pd.read_csv(file)
    PPG = DF.PPG.values[0:]
    PPG_array.append(PPG)
    SPG = DF.SPG.values[0:]
    SPG_array.append(SPG)

    # Create array of subject number and fps
    head, tail = os.path.split(file)
    regex = re.compile(r'\d+')
    NumArray = regex.findall(tail)
    SubjectList.append(NumArray[0])
    fpsList.append(NumArray[1])

    ## HR extraction ##
    L = len(SPG)
    WindowSize = 15
    fs = 41 * 2.5
    overlap = 1
    lim1 = 20
    lim2 = 200
    fftlength = np.fix(WindowSize * fs).__int__()
    NumFramesOverlap = np.fix(overlap * fs).__int__()
    fftlength2 = pow(2, np.floor(np.log(fftlength * 10) / np.log(2))).__int__()
    f = process.getFreq(fftlength2, fs)
    RangeFreq = np.where((f * 60 >= lim1) & (f * 60 <= lim2))[0]
    f2 = f[RangeFreq]
    time = np.linspace(0, L / fs, L)

    # Sliding Window

    PPGR2 = []
    SPGR2 = []
    SpectrumSPGR = []
    SpectrumPPGR = []
    ACDCPPGR = []
    ACDCSPGR = []
    SNRPPGR = []
    SNRSPGR = []

    Filter = True

    for i in range(L):
        PPGR2.append(PPG[i])
        SPGR2.append(SPG[i])

        if len(PPGR2) == fftlength:
            YRef = process.getSpectrum(SPGR2, fftlength2, f, filter=True, lim1=lim1, fs=fs)

            Spectrum = process.getSpectrum(SPGR2, fftlength2, f, filter=True, lim1=lim1, fs=fs)
            SNRSPGR.append(process.getSNR(Spectrum, f, RangeFreq, fs))
            Spectrum = process.getSpectrum(SPGR2, fftlength2, f, filter=False, lim1=lim1, fs=fs)
            SpectrumSPGR.append(Spectrum[RangeFreq])

            Spectrum = process.getSpectrum(PPGR2, fftlength2, f, filter=True, lim1=lim1, fs=fs)
            SNRPPGR.append(process.getSNR(Spectrum, f, RangeFreq, fs))
            Spectrum = process.getSpectrum(PPGR2, fftlength2, f, filter=False, lim1=lim1, fs=fs)
            SpectrumPPGR.append(Spectrum[RangeFreq])

            del SPGR2[:NumFramesOverlap]
            del PPGR2[:NumFramesOverlap]

    # fig, axes = plt.subplot_mosaic("AB;IL", sharex=True)
    # fig.set_tight_layout(True)
    # # Set general font size
    # fig.set_size_inches(200, 200)
    # axes['A'].plot(time, PPG, 'C3')
    # axes['I'].plot(time, SPG, 'C0')
    # axes['A'].set_title('Raw Signals')
    # axes['I'].legend(['PPG'])
    # axes['I'].legend(['SPG'])
    # axes['A'].legend(['PPG'])
    # axes['A'].set_ylabel("Avg. frame intensity", fontsize=8)
    # axes['I'].set_xlabel("Time (s)", fontsize=8)
    # axes['I'].set_ylabel("Avg. spatial st. dev.", fontsize=8)
    # axes['A'].set_ylabel("Avg. frame intensity", fontsize=8)
    # axes['A'].yaxis.set_tick_params(labelsize=8)
    # axes['I'].yaxis.set_tick_params(labelsize=8)
    # axes['I'].xaxis.set_tick_params(labelsize=8)
    #
    # axes['B'].set_title('Short Time Fourier Transform')
    # axes['B'].pcolor(time[0:len(SpectrumPPGR) * NumFramesOverlap:NumFramesOverlap], f2 * 60,
    #                  np.transpose(SpectrumPPGR / np.max(SpectrumPPGR)))
    # axes['B'].set_ylim(lim1, lim2)
    # axes['B'].set_ylabel("Frequency (BPM)", fontsize=8)
    #
    # axes['L'].pcolor(time[0:len(SpectrumSPGR) * NumFramesOverlap:NumFramesOverlap], f2 * 60,
    #                  np.transpose(SpectrumSPGR / np.max(SpectrumSPGR)))
    # axes['L'].set_ylim(lim1, lim2)
    # axes['L'].set_ylabel("Frequency (BPM)", fontsize=8)
    # axes['B'].yaxis.set_tick_params(labelsize=8)
    # axes['L'].yaxis.set_tick_params(labelsize=8)
    # axes['L'].xaxis.set_tick_params(labelsize=8)
    # #axes['L'].set_xlabel("Time (s)", fontsize=8)
    #
    # plt.show()
    #

### Subject 1 ###
# fig, axs = plt.subplots(2, 1, layout='constrained', figsize=(11, 5))
# b, a = butter(5, [30 / 60, 250 / 60], btype='bandpass', fs=100)
# time = np.linspace(0, 20, 100*20)
# axs[0].plot(time, SPG_array[0], '#FC5A50') #filtfilt(b, a, SPG_array[0])
# axs[0].plot(time, SPG_array[1], '#069AF3')
# axs[0].plot(time, SPG_array[2], '#15B01A')
#
# axs[1].plot(time, PPG_array[0], '#FC5A50')
# axs[1].plot(time, PPG_array[1], '#069AF3')
# axs[1].plot(time, PPG_array[2], '#15B01A')
#
# axs[0].set_ylabel(r'$SPG$ [s]')
# axs[1].set_xlabel(r'$Time$ [s]')
# axs[1].set_ylabel(r'$PPG$ [s]')
# axs[0].set_xlim([0, 20])
# axs[1].set_xlim([0, 20])
# axs[0].legend(['Subject ' + SubjectList[0] + ' ' + fpsList[0] + 'kHz', 'Subject ' + SubjectList[1] + ' ' + fpsList[1] + 'kHz', 'Subject ' + SubjectList[2] + ' ' + fpsList[2] + 'kHz'])
#
#
# ## Subject 2 ###
# fig, axs = plt.subplots(2, 1, layout='constrained', figsize=(11, 5))
# axs[0].plot(time, SPG_array[3], '#FC5A50')
# axs[0].plot(time, SPG_array[4], '#069AF3')
# axs[0].plot(time, SPG_array[5], '#15B01A')
#
# axs[1].plot(time, PPG_array[3], '#FC5A50')
# axs[1].plot(time, PPG_array[4], '#069AF3')
# axs[1].plot(time, PPG_array[5], '#15B01A')
#
# axs[0].set_ylabel(r'$SPG$ [s]')
# axs[1].set_xlabel(r'$Time$ [s]')
# axs[1].set_ylabel(r'$PPG$ [s]')
# axs[0].set_xlim([0, 20])
# axs[1].set_xlim([0, 20])
# axs[0].legend(['Subject ' + SubjectList[3] + ' ' + fpsList[3] + 'kHz', 'Subject ' + SubjectList[4] + ' ' + fpsList[4] + 'kHz', 'Subject ' + SubjectList[5] + ' ' + fpsList[5] + 'kHz'])
#
# plt.show()




### Load g2 function ###
folder = r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR_experiment\g2_maps\\' + Position + '\\Short lag'
#folder=r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR_experiment\g2_maps\Neck\3kHz data'
files = glob.glob(os.path.join(folder, '*.npy'))
final_fps = 100
T = 0.1
print(files)

mean_g2 = []
SubjectList = []
fpsList = []
for file in files:
    print(file)
    # Load the g2_map
    g2_map = np.load(file)

    # Plot test
    # plt.plot()
    # plt.plot(g2_map[100])
    # plt.show()

    # Average 2 seconds of g2 functions for log plot #
    # Subject 1 3kHz Neck
    #temp = np.mean(g2_map[80000:80000+2*final_fps], axis=0)
    temp = np.mean(g2_map[0:2*final_fps], axis=0)
    mean_g2.append(temp)

    # Create array of subject number and fps
    head, tail = os.path.split(file)
    regex = re.compile(r'\d+')
    NumArray = regex.findall(tail)
    SubjectList.append(NumArray[0])
    fpsList.append(NumArray[1])

    # Plot test
    # plt.plot()
    # plt.plot(temp)
    # plt.show()

### Plots ###

# plt.figure()
# lag10 = np.arange(1/(float(fpsList[0]) * 1000), 0.1+1/(float(fpsList[0]) * 1000), 1/(float(fpsList[0]) * 1000))
# lag25 = np.arange(1/(float(fpsList[1]) * 1000), 0.1+1/(float(fpsList[1]) * 1000), 1/(float(fpsList[1]) * 1000))
# lag50 = np.arange(1/(float(fpsList[2]) * 1000), 0.1+1/(float(fpsList[2]) * 1000), 1/(float(fpsList[2]) * 1000))
# plt.semilogx(lag10, (mean_g2[0]-min(mean_g2[0]))/(max(mean_g2[0])-min(mean_g2[0])), '#069AF3')
# plt.semilogx(lag25, (mean_g2[1]-min(mean_g2[1]))/(max(mean_g2[1])-min(mean_g2[1])), '#069AF3', linestyle='--')
# plt.semilogx(lag50, (mean_g2[2]-min(mean_g2[2]))/(max(mean_g2[2])-min(mean_g2[2])), '#069AF3', marker='.')
# plt.semilogx(lag10, (mean_g2[3]-min(mean_g2[3]))/(max(mean_g2[3])-min(mean_g2[3])), '#FC5A50')
# plt.semilogx(lag25, (mean_g2[4]-min(mean_g2[4]))/(max(mean_g2[4])-min(mean_g2[4])), '#FC5A50', linestyle='--')
# plt.semilogx(lag50, (mean_g2[5]-min(mean_g2[5]))/(max(mean_g2[5])-min(mean_g2[5])), '#FC5A50', marker='.')
# Max = 1.05
# Min = 0.99
# plt.vlines(166 * 10 ** -6, 0, 1, 'r')
# # if Position == 'Finger_top' or Position == 'Finger_bottom':
# #     plt.vlines(7.14 * 10 ** -5, Min, Max, 'r')
# #     #.axhspan(ymin, ymax, xmin=0, xmax=1, **kwargs)
# #     plt.vlines(0.9, Min, Max, 'r')
# # elif Position == 'Neck':
# #     plt.vlines(5.88 * 10 ** -5, Min, Max, 'r')
# #     plt.vlines(0.014, Min, Max, 'r')
# #     plt.vlines(0.005, Min, Max, 'r')
# plt.xlabel(r'$\tau$ [s]')
# plt.ylabel(r'g2($\tau$)')
# plt.legend(['Subject ' + SubjectList[0] + ' ' + fpsList[0] + 'kHz', 'Subject ' + SubjectList[1] + ' ' + fpsList[1] + 'kHz', 'Subject ' + SubjectList[2] + ' ' + fpsList[2] + 'kHz', 'Subject ' + SubjectList[3] + ' ' + fpsList[3] + 'kHz', 'Subject ' + SubjectList[4] + ' ' + fpsList[4] + 'kHz', 'Subject ' + SubjectList[5] + ' ' + fpsList[5] + 'kHz'])
# plt.show()

########################################################################################################################
########################################################################################################################

### Plot 10kHz ###
# plt.figure()
# lag10 = np.arange(1/(float(fpsList[0]) * 1000), T+1/(float(fpsList[0]) * 1000), 1/(float(fpsList[0]) * 1000))
# #plt.semilogx(lag10, mean_g2[0], '#069AF3')
# #plt.semilogx(lag10, mean_g2[3], '#FC5A50')
# plt.plot(lag10, (mean_g2[0]-min(mean_g2[0]))/(max(mean_g2[0])-min(mean_g2[0])), '#069AF3')
# plt.plot(lag10, (mean_g2[3]-min(mean_g2[3]))/(max(mean_g2[3])-min(mean_g2[3])), '#FC5A50')
# if Position == 'Finger_top' or Position == 'Finger_bottom':
#     Max = 1.025
#     Min = 0.999
#     plt.vlines(7.14 * 10 ** -5, Min, Max, 'r')
#     plt.vlines(0.03, Min, Max, 'r')
# elif Position == 'Neck':
#     Max = 1.02
#     Min = 0.999
#     plt.vlines(5.88 * 10 ** -5, Min, Max, 'r')
#     plt.vlines(0.014, Min, Max, 'r')
#     plt.vlines(0.005, Min, Max, 'r')
# plt.xlabel(r'$\tau$ [s]')
# plt.ylabel(r'g2($\tau$)')
# #plt.xlim([0, 1])
# plt.legend(['Subject ' + SubjectList[0] + ' ' + fpsList[0] + 'kHz']) #, 'Subject ' + SubjectList[3] + ' ' + fpsList[3] + 'kHz'])

### Plot 25kHz ###
# plt.figure()
# lag25 = np.arange(1/(float(fpsList[1]) * 1000), T+1/(float(fpsList[1]) * 1000), 1/(float(fpsList[1]) * 1000))
# #plt.semilogx(lag25, mean_g2[1], '#069AF3')
# #plt.semilogx(lag25, mean_g2[4], '#FC5A50')
# plt.plot(lag25, (mean_g2[1]-min(mean_g2[1]))/(max(mean_g2[1])-min(mean_g2[1])), '#069AF3')
# plt.plot(lag25, (mean_g2[4]-min(mean_g2[4]))/(max(mean_g2[4])-min(mean_g2[4])), '#FC5A50')
# if Position == 'Finger_top' or Position == 'Finger_bottom':
#     Max = 1.025
#     Min = 0.999
#     plt.vlines(7.14 * 10 ** -5, Min, Max, 'r')
#     plt.vlines(0.03, Min, Max, 'r')
# elif Position == 'Neck':
#     Max = 1.02
#     Min = 0.999
#     plt.vlines(5.88 * 10 ** -5, Min, Max, 'r')
#     plt.vlines(0.014, Min, Max, 'r')
#     plt.vlines(0.005, Min, Max, 'r')
# plt.xlabel(r'$\tau$ [s]')
# plt.ylabel(r'g2($\tau$)')
# #plt.xlim([0, 1])
# plt.legend(['Subject ' + SubjectList[1] + ' ' + fpsList[1] + 'kHz']) #, 'Subject ' + SubjectList[4] + ' ' + fpsList[4] + 'kHz'])
#


### Plot 50kHz ###
plt.figure()
ax = plt.gca()
lag50 = np.arange(1/(float(fpsList[2]) * 1000), T+1/(float(fpsList[2]) * 1000), 1/(float(fpsList[2]) * 1000))
#plt.semilogx(lag50, mean_g2[5])
plt.semilogx(lag50, mean_g2[2], '#069AF3')
#plt.plot(lag50, (mean_g2[2]-min(mean_g2[2]))/(max(mean_g2[2])-min(mean_g2[2])), '#069AF3')
#plt.scatter(lag50, mean_g2[2]) #, '#069AF3')
plt.semilogx(lag50, mean_g2[5], '#FC5A50')
#plt.plot(lag50, (mean_g2[5]-min(mean_g2[5]))/(max(mean_g2[5])-min(mean_g2[5])), '#069AF3')
if Position == 'Finger_top' or Position == 'Finger_bottom':
    Max = 1.03
    Min = 0.999
    if Brownian == True:
        plt.vlines(166 * 10 ** -6, Min, Max, 'r')
    elif AllVelocities == True:
        #
        # # Wall motion capillaries
        # ax.axvspan(1.99, 1.01, facecolor='blue', alpha=0.1)
        # ax.text(0.035, 1.02, 'WM capillaries', color='blue', fontsize=12)

        # BF capillaries
        ax.axvspan(0.014, 0.015, facecolor='violet', alpha=1)
        ax.text(0.01, 1.015, 'BF capillaries', color='violet', fontsize=12)

        # Wall motion artery
        ax.axvspan(0.029, 0.031, facecolor='blue', alpha=1)
        ax.text(0.035, 1.02, 'WM artery', color='blue', fontsize=12)

        # BF artery
        ax.axvspan(4 * 10 ** -5, 7.14 * 10 ** -5, facecolor='red', alpha=0.1)
        ax.text((7.14 * 10 ** -5) / 3, 1.015, 'BF artery', color='red', fontsize=12)

        # Brownian
        # ax.axvspan(0, 166 * 10 ** -6, facecolor='green', alpha=0.1)
        # ax.text((166 * 10 ** -6)/5, 1.010, 'Brownian', color='green', fontsize=12)
    else:
        plt.vlines(7.14 * 10 ** -5, Min, Max, 'r')
        plt.vlines(0.03, Min, Max, 'r')
elif Position == 'Neck':
    Max = 1.02
    Min = 0.999
    #BF capillaries
    ax.axvspan(0.0145, 0.015, facecolor='violet', alpha=1)
    ax.text(0.01, 1.015, 'BF capillaries', color='violet', fontsize=12)

    # Wall motion artery Long
    ax.axvspan(0.0048, 0.005, facecolor='blue', alpha=1)
    ax.text(0.001, 1.01, 'Long. WM artery ', color='blue', fontsize=12)

    # Wall motion artery Lat
    ax.axvspan(0.0135, 0.014, facecolor='green', alpha=1)
    ax.text(0.01, 1.011, 'Lat. WM artery', color='green', fontsize=12)

    # BF artery
    ax.axvspan(0, 5.88 * 10 ** -5, facecolor='red', alpha=0.1)
    ax.text((5.88 * 10 ** -5) / 3, 1.01, 'BF artery', color='red', fontsize=12)

    # if Brownian == True:
    #     plt.vlines(166 * 10 ** -6, Min, Max, 'r')
    # else:
    #     plt.vlines(5.88 * 10 ** -5, Min, Max, 'r')
    #     plt.vlines(0.014, Min, Max, 'r')
    #     plt.vlines(0.005, Min, Max, 'r')
# Slope and plateau limits
# Max = 1.03
# Min = 0.999
# Blue - Subject 1
# plt.vlines(6 * 10 ** -5, Min, Max, 'k')
# plt.vlines(0.001, Min, Max, 'k')
# plt.vlines(0.010, Min, Max, 'k')
# Red - Subject 2
# plt.vlines(6 * 10 ** -5, Min, Max, 'k')
# plt.vlines(0.0003, Min, Max, 'k')
# plt.vlines(0.004, Min, Max, 'k')
plt.xlabel(r'$\tau$ [s]')
plt.xlim([0.00002, 0.1])
plt.ylabel(r'g2($\tau$)')
#plt.xlim([1/(float(fpsList[2]) * 1000), 0.0005])
plt.legend(['Subject ' + SubjectList[2] + ' ' + fpsList[2] + 'kHz', 'Subject ' + SubjectList[5] + ' ' + fpsList[5] + 'kHz'])
#plt.legend(['Subject ' + SubjectList[5] + ' ' + fpsList[5] + 'kHz'])
plt.show()


### Plot 3kHz New Data ###
# plt.figure()
# lag3 = np.arange(1/(float(fpsList[0]) * 100), 1+1/(float(fpsList[0]) * 100), 1/(float(fpsList[0]) * 100))
# #plt.semilogx(lag25, mean_g2[1], '#069AF3')
# #plt.semilogx(lag25, mean_g2[4], '#FC5A50')
# plt.plot(lag3, mean_g2[0], '#069AF3')
# if Position == 'Finger_top' or Position == 'Finger_bottom':
#     Max = 1.025
#     Min = 0.999
#     plt.vlines(7.14 * 10 ** -5, Min, Max, 'r')
#     plt.vlines(0.9, Min, Max, 'r')
# elif Position == 'Neck':
#     Max = 1.02
#     Min = 0.999
#     plt.vlines(5.88 * 10 ** -5, Min, Max, 'r')
#     plt.vlines(0.014, Min, Max, 'r')
#     plt.vlines(0.005, Min, Max, 'r')
# plt.xlabel(r'$\tau$ [s]')
# plt.ylabel(r'g2($\tau$)')
# plt.legend(['Subject ' + SubjectList[0] + ' ' + fpsList[0] + 'kHz'])
# plt.show()

# plt.figure()
# plt.semilogx(lag10, (mean_g2[0]-min(mean_g2[0]))/(max(mean_g2[0])-min(mean_g2[0])), '#069AF3')
# plt.semilogx(lag25, (mean_g2[1]-min(mean_g2[1]))/(max(mean_g2[1])-min(mean_g2[1])), '#069AF3', linestyle='--')
# plt.semilogx(lag50, (mean_g2[2]-min(mean_g2[2]))/(max(mean_g2[2])-min(mean_g2[2])), '#069AF3', marker='.')
# plt.semilogx(lag10, (mean_g2[3]-min(mean_g2[3]))/(max(mean_g2[3])-min(mean_g2[3])), '#FC5A50')
# plt.semilogx(lag25, (mean_g2[4]-min(mean_g2[4]))/(max(mean_g2[4])-min(mean_g2[4])), '#FC5A50', linestyle='--')
# plt.semilogx(lag50, (mean_g2[5]-min(mean_g2[5]))/(max(mean_g2[5])-min(mean_g2[5])), '#FC5A50', marker='.')
# plt.xlabel(r'$\tau$ [s]')
# plt.ylabel(r'g2($\tau$)')
# plt.legend(['Subject ' + SubjectList[0] + ' ' + fpsList[0] + 'kHz', 'Subject ' + SubjectList[1] + ' ' + fpsList[1] + 'kHz', 'Subject ' + SubjectList[2] + ' ' + fpsList[2] + 'kHz', 'Subject ' + SubjectList[3] + ' ' + fpsList[3] + 'kHz', 'Subject ' + SubjectList[4] + ' ' + fpsList[4] + 'kHz', 'Subject ' + SubjectList[5] + ' ' + fpsList[5] + 'kHz'])


# plt.figure()
# plt.semilogx(lag10, (mean_g2[0]-min(mean_g2[0]))/(max(mean_g2[0])-min(mean_g2[0])), '#069AF3')
# plt.semilogx(lag25, (mean_g2[1]-min(mean_g2[1]))/(max(mean_g2[1])-min(mean_g2[1])), '#069AF3', linestyle='--')
# plt.semilogx(lag50, (mean_g2[2]-min(mean_g2[2]))/(max(mean_g2[2])-min(mean_g2[2])), '#069AF3', marker='.')
# plt.semilogx(lag10, (mean_g2[3]-min(mean_g2[3]))/(max(mean_g2[3])-min(mean_g2[3])), '#FC5A50')
# plt.semilogx(lag25, (mean_g2[4]-min(mean_g2[4]))/(max(mean_g2[4])-min(mean_g2[4])), '#FC5A50', linestyle='--')
# plt.semilogx(lag50, (mean_g2[5]-min(mean_g2[5]))/(max(mean_g2[5])-min(mean_g2[5])), '#FC5A50', marker='.')
# plt.xlabel(r'$\tau$ [s]')
# plt.ylabel(r'g2($\tau$)')
# plt.legend(['Subject ' + SubjectList[0] + ' ' + fpsList[0] + 'kHz', 'Subject ' + SubjectList[1] + ' ' + fpsList[1] + 'kHz', 'Subject ' + SubjectList[2] + ' ' + fpsList[2] + 'kHz', 'Subject ' + SubjectList[3] + ' ' + fpsList[3] + 'kHz', 'Subject ' + SubjectList[4] + ' ' + fpsList[4] + 'kHz', 'Subject ' + SubjectList[5] + ' ' + fpsList[5] + 'kHz'])
# plt.show()


# plt.figure()
# lag10 = np.arange(0, 0.1, 1/(float(fpsList[0]) * 1000))
# lag25 = np.arange(0, 0.1, 1/(float(fpsList[1]) * 1000))
# lag50 = np.arange(0, 0.1, 1/(float(fpsList[2]) * 1000))
# plt.semilogy(lag10, mean_g2[0], '#069AF3')
# plt.semilogy(lag25, mean_g2[1], '#069AF3', linestyle='--')
# plt.semilogy(lag50, mean_g2[2], '#069AF3', marker='.')
# plt.semilogy(lag10, mean_g2[3], '#FC5A50')
# plt.semilogy(lag25, mean_g2[4], '#FC5A50', linestyle='--')
# plt.semilogy(lag50, mean_g2[5], '#FC5A50', marker='.')
# plt.xlabel(r'$\tau$ [s]')
# plt.ylabel(r'g2($\tau$)')
# plt.legend(['Subject ' + SubjectList[0] + ' ' + fpsList[0] + 'kHz', 'Subject ' + SubjectList[1] + ' ' + fpsList[1] + 'kHz', 'Subject ' + SubjectList[2] + ' ' + fpsList[2] + 'kHz', 'Subject ' + SubjectList[3] + ' ' + fpsList[3] + 'kHz', 'Subject ' + SubjectList[4] + ' ' + fpsList[4] + 'kHz', 'Subject ' + SubjectList[5] + ' ' + fpsList[5] + 'kHz'])


### Exponential fitting ###
# a_parameter = np.linspace(-1500, -1300, 1000)
#
# data = mean_g2[2]
# minima = []
# std = []
# for ai in a_parameter:
#     function = np.exp(ai*lag50)
#     temp = []
#     for i in range(len(lag10)):
#         function_norm = (function[i]-min(function))/(max(function)-min(function))
#         data_norm = (data[i]-min(data))/(max(data)-min(data))
#         temp.append(np.square(data_norm-function_norm))
#     std.append(np.sqrt(np.mean(temp)))
#
# # Check of standard deviation #
# plt.figure()
# plt.plot(a_parameter, std)
# plt.xlabel('Exponential fit coefficient')
# plt.ylabel('Std data vs model')
#
#
# plt.figure()
# fit = np.exp(-1000*lag50)
# plt.plot((fit-min(fit))/(max(fit)-min(fit)))
# plt.plot((data-min(data))/(max(data)-min(data)))
# plt.xlabel('Exponential fit coefficient')
# plt.ylabel('Std data vs model')
# plt.show()
#
# minima.append(a_parameter[np.argmin(std)])
# print(minima)
#
# plt.figure()
# plt.plot(lag50, (mean_g2[0]-min(data))/(max(data)-min(data)), '#069AF3')
# plt.plot(lag50, np.exp(minima*lag50))
# plt.show()


