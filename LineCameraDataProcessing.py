import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import NCC_functions
import process
import pandas as pd
import pims
import h5py
import glob
from pathlib import Path

### Import NCC functions ###
c = NCC_functions.CorrelationFunction()

" This code is made to process 2 type of data:" \
"1- Seq. videos from normal camera " \
"2- Pictures from a line scan camera (global shutter)" \
"3- Matlab tensor which contains the 3kHz speckle videos aquired in Leuven"

"We several processing to the data:" \
"1- Compute the spatial g2 function" \
"2- Compute the temporal g2 function aka NCC" \
"3- Compute the standard deviation between temporal and spatial g2 function" \
"4- Compute SPG/PPG"

### Loading data ###

# Set parameters #
fps = 50000
Subject = 2
Position = 'Finger_top'
MeasurementTime = 20 # [s]
final_fps = 100
timelag = 0.1 # [s]
lag = int(timelag * fps)

# Uncomment format in which your data is writen: normal camera video are in .seq 'seq' while the linescan camera images are '.tiff'
#Format = 'seq'
Format = 'tiff'
#Format = 'mat'

# Set directory #
# Format == 'seq'
#dirname = r'X:\HFR_Pulsatrix\3170fps\315us\Neck\Subject1'
#dirname = r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR_experiment\3kHz\Sub 2 Neck'

# Format == 'tiff'
dirname = r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR_experiment\Subject ' + str(Subject) + '\\' + Position + '\\' + str(int(fps/1000)) + 'kHz'

# Format == 'mat'
#dirname = r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR'


# Extract frames #
if Format == 'seq':
    file = glob.glob(f'{dirname}/*.seq')
    final = pims.open(file[0])
    length, x_pixels, y_pixels,  = final.shape
    # Reshaping the tensor #
    finalframe = []
    for frame in final:
        finalframe.append(frame)
        # plt.figure()
        # plt.imshow(frame)
        # plt.show()

elif Format == 'tiff':
    # Init.
    final = []
    i = 0
    for fname in os.listdir(dirname):
        i = i + 1
        im = Image.open(os.path.join(dirname, fname))
        imarray = np.array(im)

        for j in range(len(imarray)):
            temp = imarray[j]
            sensor = temp[0:500]
            final.append(sensor)
        #length, pixels = imarray.shape

        # Test frame plot
        # plt.figure()
        # plt.imshow(imarray)
        # plt.show()
        # plt.figure(figsize=(18, 2))
        # plt.imshow(imarray[10:100])
        # plt.show()

    # Reshaping the frame #
    final = np.array(final).tolist()
    #finalframe = np.reshape(final, (i * length, pixels))
    finalframe = np.reshape(final, (i * len(imarray), len(sensor)))

elif Format == 'mat':
    video = h5py.File(dirname + '\\Subject2_release_6Seconds_PythonFormat.mat', 'r')
    finalframe = np.array(video['data'])

else:
    print('I do not load the format you gave me, try again!')

#
# plt.figure()
# plt.imshow(finalframe[12000]-finalframe[15000])
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(2, 1, 1)
# #Add the vmin and vmax arguments to set the color scale
# ax.imshow(finalframe[12000], vmin = 0, vmax = 255)
# #ax.set_adjustable('box-forced')
# ax.autoscale(False)
# ax2 = fig.add_subplot(2, 1, 2)
# #ax2.set_adjustable('box-forced')
# #Add the vmin and vmax arguments to set the color scale
# ax2.imshow(finalframe[30000], vmin = 0, vmax = 255)
# ax2.autoscale(False)
# plt.show()




### g2 function calculation ###

# Spatial (frame) g2 function
def frame_g2function(finalframe, lag, initialindex):
    # `frames` is a 2D array with shape (time, pixels)
    # Each column corresponds to intensity changes over time for one pixel
    g2 = np.zeros(lag)
    for i in range(lag):
        # plt.figure()
        # plt.imshow(finalframe[initialindex])
        # plt.show()
        Num = np.mean(finalframe[initialindex] * finalframe[initialindex+i])
        Den = np.mean(finalframe[initialindex]) * np.mean(finalframe[initialindex+i])
        g2[i] = Num/Den
    return g2


# Initialization array
g2_map = []
NCC_map = []
SPG = []
PPG = []
std_temp = []
Rsquared = []
for i in np.linspace(0, len(finalframe)-lag, int(MeasurementTime*final_fps)):
    i = int(i)
    print('Processing:', int(i/(len(finalframe)-lag)*100), '%')

    ### Using the spatial g2 function ###
    g2 = frame_g2function(finalframe, lag, i)
    g2_map.append(g2)
    # Plot check
    # plt.figure()
    # plt.plot(g2)
    # plt.show()

    if Format == 'seq' or Format == 'mat':

        ### Using the NCC ###
        NCC = c.compute_NCC_curve_wit_fft_tensor(finalframe[i:i + lag]) ## which dimension?
        NCC_map.append(NCC)
        # Plot check
        # plt.figure()
        # plt.plot(NCC)
        # plt.show()


    elif Format == 'tiff':

        ### Using the NCC ###
        NCC = c.compute_NCC_curve_wit_fft(finalframe[i:i+lag])
        NCC_map.append(NCC)
        # Plot check
        # plt.figure()
        # plt.plot(NCC)
        # plt.show()


    ### Calculate least square of NCC vs g2 ###
    temp1 = []
    temp2 = []
    Normg2 = (g2 - min(g2)) / (max(g2) - min(g2))
    NormNCC = (NCC - min(NCC)) / (max(NCC) - min(NCC))
    Meang2 = np.mean(Normg2)
    for k in range(len(NCC)):
        #var.append(np.square(Normg2[k] - NormNCC[k]))
        temp1.append(np.square(Normg2[k]-NormNCC[k]))
        temp2.append(np.square(Normg2[k]-Meang2))
    SSres = np.sum(temp1)
    SStot = np.sum(temp2)
    Rsquared.append(1-SSres/SStot)

    # Test plot
    # plt.figure()
    # plt.plot(NCC)
    # plt.figure()
    # Normg2 = (g2-min(g2))/(max(g2)-min(g2))
    # NormNCC = (NCC-min(NCC))/(max(NCC)-min(NCC))
    # plt.plot(Normg2-NormNCC)
    # plt.show()

    # Calculate SPG and PPG
    Frame = finalframe[i]
    temp = process.getSPG(Frame, method='sigma')
    SPG.append(temp)
    temp = Frame[np.isfinite(Frame)].mean()
    PPG.append(temp)

df = pd.DataFrame({'SPG': SPG, 'PPG': PPG})

# The standard deviation for this data set is:
std = np.mean(Rsquared)
print('Stand. dev. g2 and NCC:', std)


# plt.figure()
# plt.plot(SPG)
# plt.show()
#
# plt.figure()
# plt.plot(g2_map[10])
# plt.plot(NCC_map[10])
# plt.show()

### Saving the data ###

# New data
#df.to_csv(r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR_experiment\SPG\\' + str(Position) + '\Subject' + str(Subject) + str(Position) + str(int(fps/1000)) + '.csv')
np.save(r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR_experiment\g2_maps\\' + str(Position) + '\Subject' + str(Subject) + str(Position) + str(int(fps/1000)) + 'kHz_PartSensor2.npy', g2_map)
#np.save(r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR_experiment\NCC_maps\\' + str(Position) + '\Subject' + str(Subject) + str(Position) + str(int(fps/1000)) + 'kHz_1sLag.npy', NCC_map)
#np.save(r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR_experiment\std\\' + str(Position) + '\Subject' + str(Subject) + str(Position) + str(int(fps/1000)) + 'kHz_1sLag.npy', std)

# Old 3kHz data
# df.to_csv(r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR_experiment\SPG\\' + str(Position) + '\OldData_Subject' + str(Subject) + str(Position) + str(int(fps/1000)) + '.csv')
# np.save(r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR_experiment\g2_maps\\' + str(Position) + '\OldData_Subject' + str(Subject) + str(Position) + str(int(fps/1000)) + 'kHz_1sLag.npy', g2_map)
# np.save(r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR_experiment\NCC_maps\\' + str(Position) + '\OldData_Subject' + str(Subject) + str(Position) + str(int(fps/1000)) + 'kHz_1sLag.npy', NCC_map)
# np.save(r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR_experiment\std\\' + str(Position) + '\OldData_Subject' + str(Subject) + str(Position) + str(int(fps/1000)) + 'kHz_1sLag.npy', std)
