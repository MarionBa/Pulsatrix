""""
Basic functions for processing
@author Ilde Lorato  ilde.lorato@imec.nl
"""

from statistics import mean

import matplotlib.pyplot as plt

#from FD_CHIMP_ECG_PPG_PY.python.bbees.chimpppgfiducialpoints import ChimpPpgFiducialPoints
import scipy.signal
import skimage
from cv2 import blur, boxFilter
import cv2
from numpy import sqrt,arange,mod,hanning,argmax
import numpy as np
from numpy.fft import fft,ifft
from scipy.signal import butter, filtfilt
import pandas as pd
import time

def getSPG(Frame, method='sigma', ksize=7):
    """"
    getSPG analyzes a single frame and returns the speckle contrast value. The kernel
    size can be specified in ksize and the method for spatial speckle contrast calculation can be specified.

    :param Frame: single frame to be analyzed (NxM)
    :param method: string to specify the method for speckle contrast calculation
    Options available:
    '2dsd_divided2d' : 2d st deviation divided by 2d average of frame
    '2dsd_divided' : 2d st deviation divided by average of full frame
    '2dsd' : 2d st deviation not divided
     :param ksize: kernel size for speckle contrast calculation [kxk]
    :return: spg (single point of the spg signal)
    """
    Frame=Frame.astype(np.float64)
    A = blur(Frame, (ksize, ksize))
    A2 = blur(Frame**2, (ksize, ksize))
    if method == 'kernelNormalized':
        V = sqrt(A2 - A ** 2)/A
    elif  method == 'frameNormalized':
        V = sqrt(A2 - A ** 2)/Frame[np.isfinite(Frame)].mean()
    elif method == 'sigma':
        V = sqrt(A2 - A ** 2)
    else:
        V = sqrt(np.abs(A2 - A ** 2))/A

    spg = V[np.isfinite(V)].mean()
    return spg


def getSPGPPGMask(Frame, mask, ksize=7):

    Frame=Frame.astype(np.float64)
    A = blur(Frame, (ksize, ksize))
    A2 = blur(Frame**2, (ksize, ksize))

    V = sqrt(A2 - A ** 2)
    Frame=Frame*mask
    V=V*mask
    # import matplotlib.pyplot as plt
    # plt.imshow(Frame)
    # plt.show()
    spg = V[np.isfinite(V)].mean()
    ppg=Frame[np.isfinite(Frame)].mean()
    return spg,ppg

def getTempSpeckle(Frames):
    """
     getTempSpeckle analyzes a multiple frames and returns the temporal speckle contrast value. This is calculated as the
     standard deviation in time for each pixel, then average the resulting standard deviation map.
     The number of frames given as input should be equal to the kernel size chosen for the temporal calculation

     :param Frames: video segment, the numer of frames will correspond to the kernel size for the temporal speckle
     contrast calculation
     :return: TSC: Temporal Speckle Contrast (single point of the temporal spg)
     """
    temp=np.std(Frames,0)
    speckle = temp[np.isfinite(temp)].mean()
    return speckle

def getFreq(length, fs):
    """
    Maps the frequency vector for fft.

    :param length: the length of the signal (if no zeropadding) or the length of the zeropadded signal
    :param fs: sampling frequency in Hz
    :return: frequency vector in Hz
    """
    if mod(length, 2) == 1:
        f= arange(0,(fs / 2),fs/length)
    else:
        f=arange(0,(fs / 2) - (fs / length),fs/length)
    return f

def getOFSpeckle(Frames, s, fs_exp = 100):
    """

    :param Frames:
    :param s: settings related to frames, including NumFrames and timestamps
    :param fs_exp: expected frequency of dataframes
    :return:
    """
    motion = []
    SPGof = []
    SPGmagnitude = []
    timestamps2 = s.timestamps
    timestamps = []
    fs_thr = 0.8 * fs_exp  # originally 80 with expected frequency 100

    for i in range(s.NumFrames):
        timestamp = str(timestamps2[i][0])
        # check if a double frame is detected; in that case add 0.01
        if (len(timestamps) > 1) and ((float(timestamp) - timestamps[-1]) == 0):
            timestamps.append(float(timestamp) + 0.01)  # values dependent on frequency
        else:
            timestamps.append(float(timestamp))

        flow = cv2.calcOpticalFlowFarneback(Frames[i - 1], Frames[i], None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        # TODO extract x, y from flow -> adjust x, y by mean displacement x, y
        meanX = np.mean(flow[..., 0])
        meanY = np.mean(flow[..., 1])

        flowCorrectXY = np.zeros(np.shape(flow))
        flowCorrectXY[..., 0] = flow[..., 0] - meanX
        flowCorrectXY[..., 1] = flow[..., 1] - meanY

        magnitudeCorrected, angleCorrected = cv2.cartToPolar(flowCorrectXY[..., 0], flowCorrectXY[..., 1])
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # calculate mean motion per frame
        meanMagnitude = np.mean(magnitude)  # test if this works or directive x, y is necessary

        motion.append(meanMagnitude)

        # extract mean motion per frame to obtain noise from blood pulse
        SPGof.append(np.mean(magnitudeCorrected))
        SPGmagnitude.append(np.mean(magnitude - meanMagnitude))

        # Check frame loss
        if (len(timestamps) > 1) and ((timestamps[-1] - timestamps[-2]) != 0) and (
                1 / (timestamps[-1] - timestamps[-2])) < fs_thr:
            print('Frame Loss Correction Activated')
            print(1 / (timestamps[-1] - timestamps[-2]))
            # Frame loss detected, append an additional frame.
            motion.append(meanMagnitude)
            timestamps.append(float(timestamp))

            # extract mean motion per frame to obtain noise from blood pulse
            SPGmagnitude.append(np.mean(magnitude - meanMagnitude))
            SPGof.append(np.mean(magnitudeCorrected))

    timestamps = np.asarray(timestamps)
    timestamps = timestamps - timestamps[0]

    return motion, SPGof, SPGmagnitude, timestamps

def getSPGfromcombinedFFT(SPGlist, fs):
    combinedFFT = []
    combinedFFT2 = []

    for subSPG in SPGlist:
        tempFreq = getFreq(len(subSPG), fs)
        fftlength = len(tempFreq)  # np.fix(WindowSize * fs)
        fftlength2 = int(pow(2, np.floor(np.log(fftlength * 10) / np.log(2))))  #zeropadding

        fft = getSpectrum(subSPG, fftlength2, tempFreq, filter=False, lim1=30, fs=fs)
        # add frequency to each other per subframe; issue = getFreq needs full time SPG
        combinedFFT.append(fft)

        fft2 = np.fft.fft(subSPG, fftlength2)
        npfreq = np.fft.fftfreq(fftlength2)
        combinedFFT2.append(fft2)

    # convert combined FFT to SPG
    #SPG = inversefft
    return combinedFFT, tempFreq, combinedFFT2, npfreq

def getHarmonics(Y,f,RangeFreq,w=5):
    M = argmax(Y[RangeFreq]) + RangeFreq[0]
    H1=np.max(Y[M-w:M+w])
    H2=np.max(Y[M*2 - (w+2):M*2 + (w+2)])
    H3=np.max(Y[M*3 - (w+4):M*3 + (w+4)])
    # import matplotlib.pyplot as plt
    # plt.figure(2)
    # plt.plot(f*60,Y)
    # plt.plot(f[M-w:M+w]*60,Y[M-w:M+w])
    # plt.plot(f[M*2 - (w):M*2 + (w)]*60,Y[M*2 - (w):M*2 + (w)])
    # plt.plot(f[M*3 - (w):M*3 + (w)]*60,Y[M*3 - (w):M*3 + (w)])
    # plt.xlim([0,20*60])
    # plt.show()
    return H1,H2,H3

def getPeaksValleys(signal,rpeaks):
    signal1d=np.diff(signal)
    for count, rr in enumerate(rpeaks[:-1]):
        if count>0:
            peaks1d.append(rr + np.argmax(signal1d[rr:rpeaks[count + 1]]))
        else:
            peaks1d = [rr + np.argmax(signal1d[rr:rpeaks[count + 1]])]
    signal2d=np.diff(np.diff(signal))
    rpeaks = rpeaks[:-1]
    for count, (jj, kk) in enumerate(zip(rpeaks, peaks1d)):
        if count>0:
            if jj == kk:
                peaks2d.append(jj)
            else:
                peaks2d.append(jj + np.argmax(signal2d[jj:kk]))
        else:
            if jj == kk:
                peaks2d = [jj]
            else:
                peaks2d = [jj + np.argmax(signal2d[jj:kk])]
    valleys = peaks2d

    for count, rr in enumerate(valleys[:-1]):
        if count>0:
            peaks.append(rr + np.argmax(signal[rr:valleys[count + 1]]))
        else:
            peaks = [rr + np.argmax(signal[rr:valleys[count + 1]])]

    valleys = peaks2d[1:]
    PAT=(peaks2d[1:]-rpeaks[1:])

    return peaks,valleys,PAT,peaks1d

def getPeaksValleys2(signal,rpeaks):
    """
       This function returns the location of Peaks, Valleys, and Upstrokes. Requires the R peaks of the ECG to function.
       It is based on the standard python function for peak detection combined with the use of the R peaks of a synced ECG
       and with the use of the 1st and 2nd derivative of the PPG/SPG signals.

       :param signal: segment of the PPG/SPG signal. Better to apply some filtering since this method uses derivatives
       (will amplify high frequency noise). We suggest bandpass filtering. Note: invert the signal if necessary to obtain
       the correct peaks and valleys (otherwise they will be inverted).
       :param rpeaks: the R peaks of a synced ECG signal
       :return: peaks, valleys, upstroke: location of the fiducial points
       """
    signal1d=np.diff(signal)
    for count, rr in enumerate(rpeaks[:-1]):
        if count>0:
            peaks1d.append(rr + np.argmax(signal1d[rr:rpeaks[count + 1]]))
        else:
            peaks1d = [rr + np.argmax(signal1d[rr:rpeaks[count + 1]])]
    signal2d=np.diff(np.diff(signal))
    rpeaks = rpeaks[:-1]
    for count, (jj, kk) in enumerate(zip(rpeaks, peaks1d)):
        if count>0:
            if jj == kk:
                peaks2d.append(jj)
            else:
                peaks2d.append(jj + np.argmax(signal2d[jj:kk]))
        else:
            if jj == kk:
                peaks2d = [jj]
            else:
                peaks2d = [jj + np.argmax(signal2d[jj:kk])]
    valleys = peaks2d

    for count, rr in enumerate(valleys[:-1]):
        if count>0:
            peaks.append(rr + np.argmax(signal[rr:valleys[count + 1]]))
        else:
            peaks = [rr + np.argmax(signal[rr:valleys[count + 1]])]


    return peaks,valleys,peaks1d

def getHarmonicsperBeat(beatsCollection,fs=40,inv=True):
    H1=[]
    H2=[]
    H3=[]
    if inv:
        mul=-1
    else:
        mul=1
    for i in range(len(beatsCollection)):
        beats = mul*beatsCollection[i]
        NN=10
        for kk in range(NN):
            beats = np.append(beats, mul*beatsCollection[i])
        f = getFreq(len(beats), fs)
        Y = getSpectrum(beats, len(beats), f, fs=fs)[:int(len(f))]
        RangeFreq = np.where((f * 60 >= 40) & (f * 60 <= 110))[0]
        try:
            H1temp, H2temp, H3temp = getHarmonics(Y, f, RangeFreq)
            # if H3temp / H1temp < 1.5:
            H1.append(H1temp)
            H2.append(H2temp)
            H3.append(H3temp)
        except:
            pass
    return H1,H2,H3

def getSpectrum(buffer,fftlength2,f, filter= False, lim1=30, fs=40):
    """
    Computes fft with zeropadding and hanning windowing. Filtering optional

    :param buffer: the signal segment to transform (raw)
    :param fftlength2: the length of the signal (if no zeropadding wanted) or a longer length if zeropadding wanted
    :param f: frequency vector in Hz, obtained with getFreq
    :param filter: True or False depending on wanted filtering (highpass will be applied in case)
    :param lim1: cutoff frequency for the highpass filter in beats/min or breaths/min
    :param fs: sampling frequency in Hz
    :return: Spectrum (only positive frequencies)
    """
    if filter:
        b, a = butter(5, [lim1 / 60], btype='highpass', fs=fs)
        buffer = filtfilt(b, a, buffer)
        buffer = buffer - mean(buffer)
    else:
        buffer = buffer - mean(buffer)
    Y = abs(fft(buffer*hanning(len(buffer)), fftlength2))[:len(f)]
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(f*60, Y)
    # plt.xlabel('Frequency (BPM)')
    # plt.show()
    return Y

def getInverseSpectrum(fft, f):
    n = len(fft)
    signal = ifft(fft)
    return signal

def getSpectrumNew(buffer,fftlength2,f,  Yref, filter= False, lim1=40, lim2=150, fs=35):
    """"
    Computes fft with zeropadding and hanning windowing. Filtering tightly around a reference spectrum optional
    """
    if filter:
        M = argmax(Yref)
        hr = f[M] * 60
        b, a = butter(5, [(hr-5) / 60, (hr+5) / 60], btype='bandpass', fs=fs)
        buffer = filtfilt(b, a, buffer)
        buffer = buffer - mean(buffer)
    else:
        buffer = buffer - mean(buffer)

    Y = abs(fft(buffer*hanning(len(buffer)), fftlength2))[:len(f)]
    return Y

def getSPGlistfromSubFrames(Frame, spgmethod, rows=2, columns=2, cropx=2, cropy=2):
    """
    Gets a list of the SPG value for subframes of the full Frame. Subframes are organized by column + row. SPG values
    are calculated per Frame.
    :Frame: to obtain SPG signal from, will be divided in subframes given bij rows, columns
    :spgmethod: SPG method to use as defined in process.getSPG
    :rows: number of rows to divide Frame in
    :columns: number of columns to divide Frame in
    :returns: np.array of SPG signal with shape (rows*columns, 1); you can stack frames by np.append(all,listSPG,axis=1)
    """
    # Possible considerations of improvement: add option to overlap frames

    # calculate number of subframes in frame
    listSPG = np.empty(shape=(rows,columns), dtype='object')
    listPPG = np.empty(shape=(rows,columns), dtype='object')

    # determine subframe size
    sizeX = Frame.shape[1]/columns
    sizeY = Frame.shape[0]/rows
    # loop through subframes
    for row in range(rows):
        for column in range(columns):
            # Option to crop the frame
            subframe = Frame[int(np.floor(sizeY*row))+cropy:int(np.floor(sizeY*(row+1)))-cropy,
                       int(np.floor(sizeX*column))+cropx:int(np.floor(sizeX*(column+1)))-cropx]

            #sweep through kernels to obtain SPG, use getSPG
            listSPG[row,column] = getSPG(subframe, method=spgmethod, ksize=7)
            listPPG[row,column] = subframe[np.isfinite(subframe)].mean()

            del(subframe)

    listSPG = np.resize(listSPG,(rows*columns, 1))
    listPPG = np.resize(listPPG, (rows * columns, 1))
    return listSPG, listPPG

def getHR(spectrum,RangeFreq, f2):
    """"
    Returns HR as maximum of spectrum in BPM
    """
    M=argmax(spectrum[RangeFreq])
    hr = f2[M] * 60
    return hr

def getSNR2(Y,f, RangeFreq, fs=40, w=6,h=1,YRef=[]):
    """
        Similar to getSNR but in this version a reference spectrum can be given.
       Calculated SNR in the frequency domain - suitable for "periodic" signals.
       This function wil buiild a binary template around the spectrum. The template will have a value
       equal to 1 for parts of the spectrum considered as signal (harmonics) and zero for parts considered as noise.
       In this function the fundamental and harmonics frequencies willbe found based on the reference spectrum.
       Suitable for noisy signals.

       :param Y: spectrum, obtainable with getSpectrum
       :param f: requency vector in Hz, obtainable with getFreq
       :param RangeFreq: range of frequencies of interest, obtainiable as
                   RangeFreq = np.where((f * 60 >= lim1) & (f * 60 <= lim2))[0]
       :param fs:  sampling frequency in Hz
       :param w: width of the template built around the fundamental (requires tuning)
       :param h: umber of harmonics to be considered in the signal part (this count is excluding the fundamental)
       :param YRef: reference spectrum (example spectrum of a reference signal), obtainable with getSpectrum
       :return: SNR (in dB)

       Examples and details can be found here:
       [1] Herranz Olazabal, Jorge, et al. "Comparison between Speckle Plethysmography and Photoplethysmography
        during Cold Pressor Test Referenced to Finger Arterial Pressure." Sensors 23.11 (2023): 5016.
       [2] De Haan, Gerard, and Vincent Jeanne. "Robust pulse rate from chrominance-based rPPG." IEEE Transactions
       on Biomedical Engineering 60.10 (2013): 2878-2886.
       """
    Y = np.asarray(Y)
    if fs>30:
        Y=Y[1:np.where(f*60>=800)[0][0]]
        YRef=YRef[1:np.where(f*60>=800)[0][0]]

    U=np.zeros(len(Y))
    if len(YRef) == 0:
        M = argmax(Y[RangeFreq]) + RangeFreq[0]
    else:
        M=argmax(YRef[RangeFreq]) + RangeFreq[0]
    U[M-w:M+w]=1
    for i in range(h):
        w=w+2
        U[M*(i+2) - w:M*(i+2) + w] = 1

    SNR=10*np.log10(sum(U*Y**2)/sum((1-U)*Y**2))
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(f[1:np.where(f*60>=800)[0][0]]*60,Y/max(Y))
    # plt.xlabel('Frequency (BPM)')
    # plt.ylabel('Normalized Spectrum amplitude')
    # # plt.plot(f[1:np.where(f*60>=800)[0][0]]*60,U)
    # plt.title('{}'.format(SNR))
    # plt.show()
    return SNR

def getSNR(Y,f, RangeFreq, fs=41, w=8,h=3):
    """
      Calculated SNR in the frequency domain - suitable for "periodic" signals.
      This function wil buiild a binary template around the spectrum. The template will have a value
      equal to 1 for parts of the spectrum considered as signal (harmonics) and zero for parts considered as noise.

      :param Y: spectrum, obtainable with getSpectrum
      :param f: frequency vector in Hz, obtainable with getFreq
      :param RangeFreq: range of frequencies of interest, obtainiable as
                  RangeFreq = np.where((f * 60 >= lim1) & (f * 60 <= lim2))[0]
      :param fs: ampling frequency in Hz
      :param w: width of the template built around the fundamental (requires tuning)
      :param h: number of harmonics to be considered in the signal part (this count is excluding the fundamental)
      :return: SNR (in dB)

      Examples and details can be found here:
      [1] Herranz Olazabal, Jorge, et al. "Comparison between Speckle Plethysmography and Photoplethysmography
       during Cold Pressor Test Referenced to Finger Arterial Pressure." Sensors 23.11 (2023): 5016.
      [2] De Haan, Gerard, and Vincent Jeanne. "Robust pulse rate from chrominance-based rPPG." IEEE Transactions
      on Biomedical Engineering 60.10 (2013): 2878-2886.
      """
    Y = np.asarray(Y)
    if fs>=30:
        Y=Y[1:np.where(f*60>=800)[0][0]]
    U=np.zeros(len(Y))
    M = argmax(Y[RangeFreq])+RangeFreq[0]
    U[M-w:M+w]=1
    for i in range(h):
        w=w+2
        U[M*(i+2) - w:M*(i+2) + w] = 1

    SNR=10*np.log10(sum(U*Y**2)/sum((1-U)*Y**2))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(f[1:np.where(f*60>=800)[0][0]]*60,Y/np.max(Y))
    plt.xlabel('Frequency (BPM)')
    plt.ylabel('Normalized Spectrum amplitude')
    #plt.plot(f[1:np.where(f*60>=800)[0][0]]*60,U)
    plt.title('{}'.format(SNR))
    plt.show()
    return SNR

def get_milliNP(x, fs=40, Filter=True, SPG= False,Yref=[], f=[]):
    'Philips unit to estimate pulsatility strength of PPG based on ACDC ratio.'
    x=np.asarray(x)
    xor=x
    if Filter:
        if len(Yref)!=0:
            xmean= np.mean(x)
            M = argmax(Yref)
            hr = f[M] * 60
            b, a = butter(1, [(hr - 12) / 60, (hr + 12) / 60], btype='bandpass', fs=fs)
            x = filtfilt(b, a, x)
            x=x+xmean
        else:
            xmean= np.mean(x)
            b, a = butter(1, [0.7, 1.7], btype='bandpass', fs=fs)
            x = filtfilt(b, a, x)
            x=x+xmean
    peaks = np.asarray(scipy.signal.find_peaks(x, distance=0.6*fs)[0])
    valleys=np.asarray(scipy.signal.find_peaks(-x, distance=0.6*fs)[0])
    if len(peaks)<=1 | len(valleys)<=1:
        AC=0
        DC=1
    else:
        TracePeaks=np.interp(np.linspace(min([peaks[0],valleys[0]]),max([peaks[-1],valleys[-1]]),100),peaks,x[peaks])
        TraceValleys=np.interp(np.linspace(min([peaks[0],valleys[0]]),max([peaks[-1],valleys[-1]]),100),valleys,x[valleys])
        AC=TracePeaks-TraceValleys
        DC=TraceValleys
    if SPG:
        acdc=np.mean(AC)*1000
    else:
        acdc=np.mean(AC / DC)*1000
    return acdc


def getACDCTrace(x, fs=40, Filter=True, Yref=[], f=[], op='interp'):
    x=np.asarray(x)
    xor=x
    b, a = butter(1, [0.6,6], btype='bandpass', fs=fs)
    x = filtfilt(b, a, x)
    b, a = butter(1, [6/60], btype='lowpass', fs=fs)
    xlp= filtfilt(b, a, x)
    peaks = np.asarray(scipy.signal.find_peaks(x, distance=0.6 * fs)[0])[1:-1]
    valleys = np.asarray(scipy.signal.find_peaks(-x, distance=0.6 * fs)[0])[1:-1]
    TracePeaks = np.interp(np.linspace(min([peaks[0], valleys[0]]), max([peaks[-1], valleys[-1]]), len(x)), peaks,
                           xor[peaks])
    TraceValleys = np.interp(np.linspace(min([peaks[0], valleys[0]]), max([peaks[-1], valleys[-1]]), len(x)), valleys,
                             xor[valleys])
    # X2=xor/TraceValleys
    # TracePeaks = np.interp(np.linspace(min([peaks[0], valleys[0]]), max([peaks[-1], valleys[-1]]), len(x)), peaks,
    #                        X2[peaks])
    # TraceValleys = np.interp(np.linspace(min([peaks[0], valleys[0]]), max([peaks[-1], valleys[-1]]), len(x)), valleys,
    #                          X2[valleys])
    # AC1=TracePeaks-1
    # AC2=TracePeaks-TraceValleys
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(X2)
    # plt.show()
    # plt.plot(peaks,x[peaks],'o')
    # plt.plot(valleys,x[valleys],'o')

    # if op in ['interp']:
    #     if Filter:
    #         if len(Yref)!=0:
    #             b, a = butter(1, [6/60], btype='lowpass', fs=fs)
    #             xmean = filtfilt(b, a, x)
    #             xmean= np.mean(x)
    #             M = argmax(Yref)
    #             hr = f[M] * 60
    #             b, a = butter(1, [(hr - 12) / 60, (hr + 12) / 60], btype='bandpass', fs=fs)
    #             x = filtfilt(b, a, x)
    #         else:
    #             xmean= np.mean(x)
    #             b, a = butter(1, [0.7, 6], btype='bandpass', fs=fs)
    #             x = filtfilt(b, a, x)
    #     else:
    #         xmean=0
    #     peaks = np.asarray(scipy.signal.find_peaks(x, distance=0.6*fs)[0])[1:-1]
    #     valleys=np.asarray(scipy.signal.find_peaks(-x, distance=0.6*fs)[0])[1:-1]
    #     x = x + xmean
    #
    #     if len(peaks)<=1 | len(valleys)<=1:
    #         AC=0
    #         DC=1
    #     else:
    #         TracePeaks=np.interp(np.linspace(min([peaks[0],valleys[0]]),max([peaks[-1],valleys[-1]]),len(x)),peaks,x[peaks])
    #         TraceValleys=np.interp(np.linspace(min([peaks[0],valleys[0]]),max([peaks[-1],valleys[-1]]),len(x)),valleys,x[valleys])
    #         AC=TracePeaks-TraceValleys
    #         DC=TraceValleys
    # else:
    #     if Filter:
    #         if len(Yref) != 0:
    #             b, a = butter(1, [6 / 60], btype='lowpass', fs=fs)
    #             xmean = filtfilt(b, a, x)
    #             xmean = np.mean(x)
    #             M = argmax(Yref)
    #             hr = f[M] * 60
    #             b, a = butter(1, [(hr - 12) / 60, (hr + 12) / 60], btype='bandpass', fs=fs)
    #             x = filtfilt(b, a, x)
    #         else:
    #             xmean = np.mean(x)
    #             b, a = butter(1, [0.7, 6], btype='bandpass', fs=fs)
    #             x = filtfilt(b, a, x)
    #     else:
    #         xmean = 0
    #     b, a = butter(1, [6 / 60], btype='lowpass', fs=fs)
    #     AC = np.abs(x)**2
    #     DC = filtfilt(b, a, x)

    acdc=(TracePeaks-TraceValleys)/TraceValleys
    # acdc=(TracePeaks-TraceValleys)

    return acdc

def getACTrace(x, fs=40, Filter=True, Yref=[], f=[]):
    """
        Returns the AC trace as the difference of the upper and lower envelopes of the PPG/SPG signal/segment. The envelopes
        are calculated based on peaks and valleys detection and then the peaks\valleys are interpolated to produce the envelopes.
        Filtering can be applied tightly around the fundamental frequency based on a reference spectrum.
        :param x: signal/segment PPG or SPG raw
        :param fs: sampling frequency in Hz
        :param Filter: True or False, filter around fundamental frequency if Yref is provided otherwise filters between
         [0.7, 6] Hz.
        :param Yref: Reference spectrum to be used for the fundamental frequency identification. Optional.
        :param f: frequency vector.
        :return: AC: AC Trace
        """
    x=np.asarray(x)
    xor=x
    if Filter:
        if len(Yref)!=0:
            b, a = butter(1, [6/60], btype='lowpass', fs=fs)
            xmean = filtfilt(b, a, x)
            # xmean= np.mean(x)
            M = argmax(Yref)
            hr = f[M] * 60
            b, a = butter(1, [(hr - 12) / 60, (hr + 12) / 60], btype='bandpass', fs=fs)
            x = filtfilt(b, a, x)
        else:
            xmean= np.mean(x)
            b, a = butter(1, [0.7, 6], btype='bandpass', fs=fs)
            x = filtfilt(b, a, x)
    else:
        xmean=0
    peaks = np.asarray(scipy.signal.find_peaks(x, distance=0.6*fs)[0])[1:-1]
    valleys=np.asarray(scipy.signal.find_peaks(-x, distance=0.6*fs)[0])[1:-1]
    x = x + xmean

    if len(peaks)<=1 | len(valleys)<=1:
        AC=0
    else:
        TracePeaks=np.interp(np.linspace(min([peaks[0],valleys[0]]),max([peaks[-1],valleys[-1]]),len(x)),peaks,x[peaks])
        TraceValleys=np.interp(np.linspace(min([peaks[0],valleys[0]]),max([peaks[-1],valleys[-1]]),len(x)),valleys,x[valleys])
        AC=TracePeaks-TraceValleys

    return AC

def getDCTrace(x, fs=40, Filter=True, Yref=[], f=[]):
    """
    Returns the DC trace as the lower envelope of the PPG/SPG signal/segment. The envelope
    is calculated based on the valleys detection and then the valleys are interpolated to produce the envelope.
    Filtering can be applied tightly around the fundamental frequency based on a reference spectrum.
    :param x: signal/segment PPG or SPG raw
    :param fs: sampling frequency in Hz
    :param Filter: True or False, filter around fundamental frequency if Yref is provided otherwise filters between
     [0.7, 6] Hz.
    :param Yref: Reference spectrum to be used for the fundamental frequency identification. Optional.
    :param f: frequency vector.
    :return: DC: DC Trace
    """
    x=np.asarray(x)
    xor=x
    if Filter:
        if len(Yref)!=0:
            b, a = butter(1, [6/60], btype='lowpass', fs=fs)
            xmean = filtfilt(b, a, x)
            # xmean= np.mean(x)
            M = argmax(Yref)
            hr = f[M] * 60
            b, a = butter(1, [(hr - 12) / 60, (hr + 12) / 60], btype='bandpass', fs=fs)
            x = filtfilt(b, a, x)
        else:
            xmean= np.mean(x)
            b, a = butter(1, [0.7, 6], btype='bandpass', fs=fs)
            x = filtfilt(b, a, x)
    else:
        xmean=0
    peaks = np.asarray(scipy.signal.find_peaks(x, distance=0.6*fs)[0])[1:-1]
    valleys=np.asarray(scipy.signal.find_peaks(-x, distance=0.6*fs)[0])[1:-1]
    x = x + xmean

    if len(peaks)<=1 | len(valleys)<=1:
        DC=1
    else:
        TraceValleys=np.interp(np.linspace(min([peaks[0],valleys[0]]),max([peaks[-1],valleys[-1]]),len(x)),valleys,x[valleys])
        DC=TraceValleys

    return DC
def getFiducialPoints(beatsCollection, fs=40, inv=True):
    Sys=[]
    Dia=[]
    Dic=[]
    SysLoc = []
    DiaLoc = []
    DicLoc = []
    if inv:
        mul = -1
    else:
        mul = 1
    for i in range(len(beatsCollection)):
        beats = mul * beatsCollection[i]
        beast2d=np.diff(np.diff(beats))
        peaksloc=scipy.signal.find_peaks(beats)[0][:2]
        Sys.append(beats[peaksloc[0]])
        SysLoc.append(peaksloc[0])
        # Dia.append(beats[peaksloc[1]])
        # DiaLoc.append(peaksloc[1])
        peaksloc=scipy.signal.find_peaks(beast2d)[0][:2]
        dloc=peaksloc[0]
        # dloc=peaksloc[0]+np.argmin(beats[peaksloc[0]:peaksloc[1]])
        Dic.append(beats[dloc])
        DicLoc.append(dloc)
    return Sys,Dia,Dic,SysLoc,DiaLoc,DicLoc

def getPulseWaveFeatures(beatsCollection,peaks, dicrot,fs=40, inv=True):
    sbp=[]
    dbp=[]
    pp=[]
    map=[]
    msbp=[]
    mdbp=[]
    esp=[]
    lvet=[]
    dt=[]
    Tp=[]
    ff=[]
    spti=[]
    dpti=[]
    sevr=[]
    if inv:
        mul = -1
    else:
        mul = 1
    for i in range(len(peaks)):
        beats = mul * beatsCollection[i]
        valley=np.argmin(beats[:peaks[i]])
        sbp.append(beats[peaks[i]])
        dbp.append(beats[valley])
        pp.append(beats[peaks[i]]-beats[valley])
        map.append(np.mean(beats))
        msbp.append(np.mean(beats[:dicrot[i]]))
        mdbp.append(np.mean(beats[dicrot[i]:]))
        esp.append(beats[dicrot[i]])
        lvet.append((dicrot[i])/fs)
        dt.append((dicrot[i]-len(beats))/fs)
        Tp.append((peaks[i])/fs)
        ff.append((map[-1]-dbp[-1])/pp[-1])
        spti.append(msbp[-1]-lvet[-1])
        dpti.append(mdbp[-1]-dt[-1])
        sevr.append(dpti[-1]/spti[-1])

    return sbp,dbp,pp,map,msbp,esp,lvet,dt,Tp,ff,spti,dpti,sevr

def getFiducialPointsCHIMP(signal,fs,time):

    beat_info = ChimpPpgFiducialPoints(signal, fs)
    py_upstroke = np.rint((beat_info[0, :] - time[0]) * fs).astype(np.int)
    py_foot = np.rint((beat_info[1, :] - time[0]) * fs).astype(np.int)
    py_peak = np.rint((beat_info[3, :] - time[0]) * fs).astype(np.int)

    return py_upstroke, py_foot,py_peak

def getUpstrokeSSF(signal,fs,w=0.05):
    b, a = butter(5, [350 / 60], btype='lowpass', fs=fs)
    s = filtfilt(b, a, signal)
    s=np.diff(s)
    x=np.where(s>0,s,s*0)
    SSF=[np.sum(x[k-int(w*fs):k]) for k in range(int(w*fs),len(x))]
    upstroke=scipy.signal.find_peaks(SSF,distance=0.4*fs)[0]

    return SSF,upstroke

def getPeaksValleysUpstrokes_Ref(signal,footRef):


    for count, rr in enumerate(footRef[:-1]):
        if count>0:
            peaks.append(rr + np.argmax(signal[rr:footRef[count + 1]]))
        else:
            peaks = [rr + np.argmax(signal[rr:footRef[count + 1]])]
    return peaks



