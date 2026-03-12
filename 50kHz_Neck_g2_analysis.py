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
from scipy.fft import fft, fftfreq
from scipy.signal import filtfilt, butter
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from matplotlib import cm
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import resample


### Load g2 function - Finger top linescan camera ###
folder = r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR_experiment\g2_maps\Neck\Short lag'
Files = list(Path(folder).glob('*.npy'))
print(Files)

# Parameters
final_fps = 100
fps = 50000
T = 20 # Total real time we want to see the signal

# Do I want a plot of the full time series (True) or select good
# waveforms for each subjects (False)
FullSeries = True

FirstSlope = False
SecondSlope = True


# Analysis
Files = [Files[2], Files[5]] # Selecting only the 50kHz data
n = 0
for file in Files:
    # Load array
    g2_map = np.load(file)
    print(file)

    # Initialization arrays
    g2_cropped = []
    g2_cropped_map = []
    g2_croppedLog = []
    g2_slice1 = []
    g2_slice2 = []
    g2_slice3 = []

    # Subject number counter
    n = n + 1

    if FullSeries == True:
        TimeSample_i = 0
        TimeSample_f = 20
    elif FullSeries == False:
        if n == 1:
            TimeSample_i = 10.5
            TimeSample_f = 12.75
        elif n == 2:
            TimeSample_i = 10.5
            TimeSample_f = 14
    ti = 10
    tf = 10.5
    for g2 in g2_map[int(ti*final_fps):int(tf*final_fps)]:
        if FirstSlope == True:
            lag_i = 0.00002
            lag_f = 0.00006

        elif SecondSlope == True:
            lag_i = 0.00008 #0.00006
            lag_f = 0.00014 #0.001

        ### Slices plot ###
        # Filtering used
        b, a = butter(5, [30 / 60, 250 / 60], btype='bandpass', fs=100)



        # g2 samples with lag time we want to select
        #g2_cropped.append(g2[int(lag_i * fps):int(lag_f * fps)])
        #g2_cropped_g2map.append(savgol_filter(g2[int(lag_i * fps):int(lag_f * fps)], window_length=10, polyorder=2))
        #g2_cropped_g2map.append(np.log(g2[int(lag_i * fps):int(lag_f * fps)]))
        g2_cropped_map.append(savgol_filter(g2[:int(0.004 * fps)], 10, 1))

        # Time arrays
        timeLag_map = np.arange(0.00002, 0.004, 1/fps)

        # Figure check
        # plt.figure()
        # plt.semilogx(timeLag, g2[int(lag_i * fps):int(lag_f * fps)])
        # plt.plot(timeLag, g2[int(lag_i * fps):int(lag_f * fps)])
        # plt.xlabel(r'$\tau$ [s]')
        # plt.ylabel(r'g2($\tau$)')
        # plt.legend(['Subject ' + str(n) + ' 50kHz'])
        # plt.show()

    # Filtering along time axis
    #g2_smoothed = savgol_filter(g2_cropped_g2map, window_length=10, polyorder=2, axis=0)

    # Time array for plots
    time = np.arange(ti, tf, 1/final_fps)

    # g2 normalization and rotation array for g2 surf map
    temp = np.rot90(np.flip(g2_cropped_map) / np.mean(g2_cropped_map))

    g2map = []
    for i in range(len(temp)-1):
        g2map.append(temp[len(temp)-1-i])

    # ### Surface plot ###
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x, y = np.meshgrid(time, timeLag_map)

    # Plot the surface.
    surf = ax.plot_surface(x, y, np.array(g2map), cmap=cm.coolwarm)
    ax.set_ylabel(r'$\tau$ [s]')
    #ax.set_xticks([10.0, 10.1, 10.2, 10.3, 10.4, 10.5])
    ax.set_xlabel(r'Time [s]')
    ax.set_zlabel(r'g2($\tau$)/$\bar{g2(0)}$')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf)
    plt.show()



    ### Fitting ###
    FitSecondSlope = 'Log' #Pol or Log or Exp or Lin
    Inflection_point = lag_f # [s]
    Coef = []
    Coef3 = []
    Coef2 = []
    Coef1 = []
    DC = []
    for g2 in g2_map[int(TimeSample_i*final_fps):int(TimeSample_f*final_fps)]:

        if FirstSlope == True:
            temp = g2[0:int(fps*Inflection_point)]
            P = np.polyfit(np.arange(1 / fps, Inflection_point+1/fps, 1 / fps), temp, 1)
            Coef.append(P[0])
            DC.append(P[1])
            # Figure check
            # plt.figure()
            # plt.plot(timeLag, temp)
            # plt.plot(timeLag, P[1] * np.exp(timeLag * P[0]))
            # plt.show()
            # if n == 2:
            #     plt.figure()
            #     plt.plot(timeLag, np.log(temp[:len(timeLag)]))
            #     plt.plot(timeLag, timeLag * P[0] + P[1])
            #     plt.show()
        elif SecondSlope == True:
            if FitSecondSlope == 'Exp':
                # Exponential fitting
                alpha = np.linspace(-5000, -5, 100)
                std = []
                for al in alpha:
                    lagTime = np.arange(lag_i, lag_f, 1 / fps)
                    exp_fit = (np.exp(al*lagTime)-min(np.exp(al*lagTime)))/(max(np.exp(al*lagTime))-min(np.exp(al*lagTime)))
                    temp = g2[int(lag_i * fps):int(lag_f * fps)]
                    norm_g2 = (temp-min(temp))/(max(temp)-min(temp))
                    # plt.figure()
                    # plt.plot(timeLag, norm_g2)
                    # plt.plot(timeLag, exp_fit)
                    # plt.legend([r'Normalized g2($\tau$)', 'Exp. fit'])
                    # plt.xlabel('Time lag [s]')
                    # #plt.ylabel(r'Normalized g2($\tau$)')
                    # plt.show()
                    var = []
                    for k in range(len(exp_fit)-1):
                        var.append(np.square(norm_g2[k]-exp_fit[k]))
                    std.append(np.sqrt(np.mean(var)))
                Coef.append(alpha[np.argmin(std)])
            elif FitSecondSlope == 'Log':
                if n == 1:
                    lagTime = np.arange(lag_i, lag_f-1/fps, 1 / fps)
                elif n == 2:
                    lagTime = np.arange(lag_i, lag_f, 1 / fps)
                temp = g2[int(lag_i * fps):int(lag_f * fps)]
                P = np.polyfit(lagTime, np.log(temp), 1)
                # plt.figure()
                # plt.plot(np.log(temp))
                # plt.plot(P[0]*lagTime+P[1])
                # plt.legend([r'log[g2($\tau$)]', 'Lin. fit'])
                # plt.xlabel('Time lag [s]')
                # #plt.xticks(False)
                # plt.show()
                Coef.append(P[0])
                DC.append(P[1])
            elif FitSecondSlope == 'Lin':
                if n == 1:
                    lagTime = np.arange(lag_i, lag_f, 1 / fps)
                elif n == 2:
                    lagTime = np.arange(lag_i, lag_f+1/fps, 1 / fps)
                temp = g2[int(lag_i * fps):int(lag_f * fps)]

                P = np.polyfit(lagTime, temp, 1)
                plt.figure()
                plt.plot(np.log(temp))
                plt.plot(P[0]*lagTime+P[1])
                plt.legend([r'log[g2($\tau$)]', 'Lin. fit'])
                plt.xlabel('Time lag [s]')
                #plt.xticks(False)
                plt.show()
                Coef.append(P[0])
            elif FitSecondSlope == 'Pol':
                if n == 1:
                    lagTime = np.arange(lag_i, lag_f, 1 / fps)
                elif n == 2:
                    lagTime = np.arange(lag_i-1/fps, lag_f, 1 / fps)
                temp = g2[int(lag_i * fps):int(lag_f * fps)]
                P = np.polyfit(lagTime, temp, 3)
                # plt.figure()
                # plt.plot(lagTime, temp)
                # plt.plot(lagTime, P[0]*np.power(lagTime, 3)+P[1]*np.power(lagTime, 2)+P[2]*np.power(lagTime, 1)+P[3])
                # plt.legend([r'g2($\tau$)', 'Pol. fit'])
                # plt.xlabel('Time lag [s]')
                # plt.show()
                Coef3.append(P[0])
                Coef2.append(P[1])
                Coef1.append(P[2])

            # if n == 1:
            #     temp = g2[int(lag_i * fps):int(lag_f * fps)]
            #     P = np.polyfit(np.arange(lag_i, lag_f, 1 / fps), temp, 1)
            # elif n == 2:
            #     temp = g2[int(lag_i * fps):int(lag_f * fps)]
            #     P = np.polyfit(np.arange(lag_i-1/fps, lag_f, 1 / fps), temp, 1)
                #P = np.polyfit(np.arange(lag_i, lag_f-1/fps, 1 / fps), temp, 1)



    ### Plot fitting coefficient time series ###

    # 2- Plot
    time = np.arange(TimeSample_i, TimeSample_f, 1 / final_fps)
    plt.figure(figsize=(10, 2))
    if FirstSlope == True:
        FiltButter_g2 = - filtfilt(b, a, Coef)
        FiltButter_DC = - filtfilt(b, a, DC)
        plt.plot(time, FiltButter_g2)
        plt.plot(time, FiltButter_DC)
        plt.ylabel('Fitting coefficient')
        plt.legend(['Inverted signal'])
    elif SecondSlope == True:
        if FitSecondSlope == 'Exp' or FitSecondSlope == 'Log':
            FiltButter_g2 = - filtfilt(b, a, Coef)
            FiltButter_DC = - filtfilt(b, a, DC)
            Sav_filt = savgol_filter(Coef, window_length=20, polyorder=2)
            #plt.plot(time, FiltButter_g2)
            plt.plot(time, FiltButter_DC*500)
            plt.ylabel('Fitting coefficient')
            plt.legend(['Inverted signal'])
        elif FitSecondSlope == 'Polynomial':

            # Savitzky- Golay filtering
            Sav_filt_coef1 = savgol_filter((Coef1-min(Coef1))/(max(Coef1)-min(Coef1)), window_length=20, polyorder=2)
            Sav_filt_coef2 = savgol_filter((Coef2 - min(Coef2)) / (max(Coef2) - min(Coef2)), window_length=20, polyorder=2)
            Sav_filt_coef3 = savgol_filter((Coef3 - min(Coef3)) / (max(Coef3) - min(Coef3)), window_length=20, polyorder=2)
            # plt.plot(time, -Sav_filt_coef3)
            # plt.plot(time, -Sav_filt_coef2 + 0.4)
            # plt.plot(time, -Sav_filt_coef1 + 0.8)

            # Bandpass filtering
            FiltButter_coef1 = filtfilt(b, a, Coef1)
            FiltButter_coef2 = filtfilt(b, a, Coef3)
            FiltButter_coef3 = filtfilt(b, a, Coef3)
            plt.plot(time, -FiltButter_coef1)
            plt.plot(time, -FiltButter_coef1 + 0.4)
            plt.plot(time, -FiltButter_coef1 + 0.8)

            # plt.plot(time, -Sav_filt)
            #plt.ylabel('Fitting coefficient')
            plt.legend(['3rd o. coef.', '2nd o. coef.', '1st o. coef.'])

    plt.xlim([TimeSample_i, TimeSample_f])
    #plt.ylim([80, 160])
    #plt.ylim([100, 160])
    plt.xlabel('Time [s]')
    plt.show()

    ### Plot spectrum ###
    # C = Coef1
    # L = len(C)
    # WindowSize = 10
    # fs = 100
    # overlap = 1
    # lim1 = 10
    # lim2 = 400
    # fftlength = np.fix(WindowSize * fs).__int__()
    # NumFramesOverlap = np.fix(overlap * fs).__int__()
    # fftlength2 = pow(2, np.floor(np.log(fftlength * 10) / np.log(2))).__int__()
    # f = process.getFreq(fftlength2, fs)
    # RangeFreq = np.where((f * 60 >= lim1) & (f * 60 <= lim2))[0]
    # f2 = f[RangeFreq]
    # print(fftlength, fftlength2, len(f))
    #
    # Spectrum = process.getSpectrum(C, fftlength2, f, filter= False, lim1=20, fs=100)
    # plt.figure()
    # plt.plot(f, Spectrum)
    # plt.show()

    # # ### Find peaks for single cardiac cycle waveform average ###
    # if n == 1:
    #     #Subject 1
    #     t_i = 10.6
    #     t_f = 12.5
    #     time = np.arange(t_i, t_f-1/final_fps, 1 / final_fps)
    #     peaks, _ = find_peaks(FiltButter_g2[int(t_i * final_fps):int(t_f * final_fps)], height=20)
    # elif n == 2:
    #     # Subject 2
    #     t_i = 10
    #     t_f = 14.5
    #     time = np.arange(t_i, t_f, 1 / final_fps)
    #     peaks, _ = find_peaks(FiltButter_g2[int(t_i * final_fps):int(t_f * final_fps)], height=30)
    #
    # # Check peaks
    # signal = FiltButter_g2[int(t_i*final_fps):int(t_f*final_fps)]
    # plt.figure()
    # plt.plot(time, signal)
    # time_peaks = []
    # signal_peaks = []
    # for i in range(len(peaks)):
    #     time_peaks.append(time[peaks[i]])
    #     signal_peaks.append(signal[peaks[i]])
    # plt.plot(time_peaks, signal_peaks, '*')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Lin. Coef.')
    # plt.show()
    #
    # # Averaged waveform (between peaks)
    # peak_array = []
    # time_bpm = []
    # for i in range(len(peaks)-1):
    #     ressampled_signal = resample(signal[peaks[i]:peaks[i+1]], 100)
    #     time_bpm.append(time[peaks[i+1]]-time[peaks[i]])
    #     print(time[peaks[i]])
    #     # plt.figure()
    #     # plt.plot(ressampled_signal)
    #     # plt.show()
    #     peak_array.append(ressampled_signal)
    # mean_waveform = np.mean(peak_array, axis=0)
    #
    # # Get heart rate
    # print('BPM [Hz]:', np.mean(time_bpm))
    # print('Time of one peak:', time[peaks[0]])

    # Get fiducials
    # Sys,Dia,Dic,SysLoc,DiaLoc,DicLoc = process.getFiducialPoints(peak_array, fs=final_fps, inv=True)
    # print(Sys,Dia,Dic)
    # ressampled_time = np.arange(0, 1, 1 / 100)
    # plt.figure()
    # for i in range(len(peak_array)):
    #     signal = peak_array[i]
    #     plt.plot(ressampled_time, signal)
    #     plt.scatter(ressampled_time[SysLoc[i]], signal[SysLoc[i]])
    # plt.legend(['Inverted signal'])
    # plt.ylabel('Lin. Coef.')
    # plt.xlabel('Time [s]')
    # plt.show()
    # Figure check waveform
    # plt.figure()
    # plt.plot(np.arange(0, 1, 1/100), -mean_waveform)
    # plt.legend(['Inverted signal'])
    # plt.ylabel('Lin. Coef.')
    # plt.xlabel('Time [s]')
    # plt.show()



    ## HR extraction ##
    C = filtfilt(b, a, Coef)
    L = len(C)
    WindowSize = 3
    fs = 100
    overlap = 0.5
    lim1 = 40
    lim2 = 100
    fftlength = np.fix(WindowSize * fs).__int__()
    NumFramesOverlap = np.fix(overlap * fs).__int__()
    fftlength2 = pow(2, np.floor(np.log(fftlength * 10) / np.log(2))).__int__()
    f = process.getFreq(fftlength2, fs)
    RangeFreq = np.where((f * 60 >= lim1) & (f * 60 <= lim2))[0]
    f2 = f[RangeFreq]
    time = np.linspace(0, L / fs, L)
    print(len(time), len(Coef))

    #Sliding Window

    Filter = True
    fitCoef = []
    SNRfitCoef = []
    SpectrumfitCoef = []



    for i in range(L):
        fitCoef.append(C[i])

        if len(fitCoef) == fftlength:

            Spectrum = process.getSpectrum(fitCoef, fftlength2, f, filter=True, lim1=lim1, fs=fs)
            SNRfitCoef.append(process.getSNR(Spectrum, f, RangeFreq, fs))
            Spectrum = process.getSpectrum(fitCoef, fftlength2, f, filter=False, lim1=lim1, fs=fs)
            SpectrumfitCoef.append(Spectrum[RangeFreq])

            del fitCoef[:NumFramesOverlap]

    fig, axes = plt.subplot_mosaic("AB", sharex=True)
    fig.set_tight_layout(False)
    # Set general font size
    fig.set_size_inches(15, 5)
    axes['A'].plot(time, C, 'C3')
    axes['A'].set_title('Signal')
    if FitSecondSlope == 'Lin':
        axes['A'].set_ylabel("Lin. coef.", fontsize=8)
    elif FitSecondSlope == 'Exp':
        axes['A'].set_ylabel("Exp. coef.", fontsize=8)
    elif FitSecondSlope == 'Pol':
        axes['A'].set_ylabel("Pol. coef.", fontsize=8)
    axes['A'].yaxis.set_tick_params(labelsize=8)
    axes['A'].set_xlabel("Time [s]", fontsize=8)
    axes['B'].set_xlabel("Time [s]", fontsize=8)
    axes['B'].set_title('Short Time Fourier Transform')
    axes['B'].pcolor(time[0:len(SpectrumfitCoef) * NumFramesOverlap:NumFramesOverlap], f2 * 60,
                     np.transpose(SpectrumfitCoef / np.max(SpectrumfitCoef))) #
    # axes['B'].pcolor(time[0:len(SpectrumfitCoef) * NumFramesOverlap:2], f2 * 60,
    #                                    np.transpose(SpectrumfitCoef / np.max(SpectrumfitCoef)))
    axes['B'].set_ylim(lim1, lim2)
    axes['B'].set_ylabel("Frequency (BPM)", fontsize=8)
    axes['B'].yaxis.set_tick_params(labelsize=8)


    plt.show()



