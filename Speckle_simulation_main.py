
""""
This code simulates a speckle video with:
-blood flow velocity
-absorption and scattering
-speckle decorrelation

@author Marion Barbeau marion.barbeau@imec.nl
"""

import numpy as np
import process
import scipy.fft
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import cv2

### ---------------------------------------###

### General parameters ###

N = 100 # Size camera sensor in pixels
Tend = 3 # Maximal time simulation
Simulation_fps = 100 # Frames per second of the simulation, has to be larger than the final_fps

time = np.arange(0,  Tend - 1 / Simulation_fps, 1 / Simulation_fps) # Time array

Final_fps = 100 # Note that for the sake of ... Simulation_fps / Final_fps = integer
Exp_time = 0.001
Conmbined_frames = Exp_time / Simulation_fps # Number of combined frames in order to create bluriness

### ---------------------------------------###

### Blood flow velocity related parameters ###

v = 200 # Blood flow velocity in mm / s
A_v = v / 2.6861 # Amplitude for velocity in mm / s
f_v = 1 # Frequency 1 Hz - heart beat

# Phase waveform
phi1_v = 0 # Phase first sinusoid
phi2_v = - 4 * np.pi / 3 # Phase second sinusoid

# Blood velocity waveform
def wave_v(x):
    return 3*A_v/4 + A_v * (np.sin(2*np.pi*(f_v * x + phi1_v)) + np.sin(4*np.pi*(f_v* x + phi2_v)))



### ---------------------------------------###

### Absorption and scattering related parameters ###

A_hct = 0.05 # Amplitude hct variation
f_hct = 1 #Frequency hct

# Phase waveform
phi1_hct = 0 # Phase first sinusoid
phi2_hct = - np.pi / 7 # Phase second sinusoid


# Absorption coefficient & Beer-Lambert law (Boschart et al.)
d = 2 # Depth, in mm
hct = 42.5 # Mean percentage of volume of red blood cells in blood, average 45% for men and 40 % for women
hct_45 = 45 # hct with known mu_a, used to derive mu_a for other hct's
f_plasma = 0.90 # Water volume fraction in blood plasma
f_RBC = 0.66 # Water volume fraction in red blood cells

# For laser wavelenght = 825 * 10 ** -6 mm
mu_a_H2O = 1.82 * 10 ** -4 # Absorption coefficient of water, in mm ^ -1
mu_a_45_oxy = 0.45 # (Oxygenized) absorption coefficient of given hct, in mm ^ -1
mu_a_45_deoxy = 0.43 #(Deoxygenized) absorption coefficient of given hct, in mm ^ -1

def f_blood(x):
    return (1 - x / 100) * f_plasma + (x / 100) * f_RBC

def Mu_a(x):
    return hct / hct_45 * (x - mu_a_H2O * f_blood(hct_45)) + mu_a_H2O * f_blood(hct)

mu_a_oxy =  Mu_a(mu_a_45_oxy)  # Oxygenized absorption coefficient
mu_a_deoxy = Mu_a(mu_a_45_deoxy) # Deoxygenized absorption coefficient
mu_a = np.mean([mu_a_oxy, mu_a_deoxy]) # Mean of oxygenized and deoxygenized absorption coefficient


# Reduced scattering coefficient - Beer Lambert Law(Boschart et al.)
R_p = 4e-3 # Particle radius(red blood cell) (in mm)
V_p = (4 / 3) * np.pi * R_p ** 3 # Particle volume, MCV(mean corpuscular volume), (in mm ^ 3)
sigma_s = np.pi * R_p ** 2 # Scattering cross section( in mm ^ 2)
mu_s = (sigma_s * (hct / 100) / V_p) # Independent scattering coefficient( in mm ^ -1)

# For laser wavelenght = 825 * 10 ** -6 mm
g = 0.9806 # Parameter for reduced scattering

mu_s_prime = mu_s * (1-g)
mu_eff = np.sqrt(3 * (mu_a * (mu_a+mu_s_prime))) # Effective scattering coefficient

# Blood volume changes waveform
def wave_hct(x):
    return -(A_hct/1.2 * (np.sin(2*np.pi*(f_hct * x + phi1_hct)) + np.sin(4*np.pi*(f_hct * x + phi2_hct))))


### ---------------------------------------###

### Absorption and scattering related parameters ###

# Parameters
P_dia = 60 # Diastolic blood pressure[mmgH]
P_sys = 110 # Systolic blood pressure[mmgH]
D_vess = 30 * 10 ** -3 # Blood vessel diameter[mm]
mu = 3e-3 * 0.0075 # Blood viscosity[mmHg.s]
E = 262.8e3 * 0.0075 # Young modulus artery 262.8\pm 99[kPa]

# Additional formula
Delta_P = P_sys - P_dia # Blood pressure difference over a cardiac cycle
radius = D_vess / 2 # Radius blood vessel
v_max = np.max(wave_v(time)) # Maximum velocity
A = np.pi * radius ** 2 # Cross sectional area
Q = A * v_max # Blood volume flow rate
delta_r = ((16 * mu * Q) / (np.pi * Delta_P)) ** (1 / 3) # Amlitude blood vessel diameter change NOP
delta_radius = Delta_P / (E * radius) # in [mm]

# Phase waveform
phi1_radius = 0 # First sinus phase
phi2_radius = 0 # Second sinus phase

# Vessel wall motion waveform
def wave_radius(x,p1,p2,p3,p4):
    return 8*p1 + p1 *(np.sin(2*np.pi*(p2 * x + p3)) + np.sin(4*np.pi*(p2 * x + p4)))


### ---------------------------------------###

### Speckle correlation time ###

cor = 0.85 # Correlation coefficient

### ---------------------------------------###

### Speckle pattern simulation ###

# Original (static) speckles
r = N / 4  # Radius
cx = N / 2 # Center x-coordinate
cy = N / 2 # Center y-coordinate


# Steps into creating the speckle pattern - Duncan algorithm
image_rand = np.sqrt((1-cor) ** 2 / (1-cor ** 2)) * (np.random.rand(N, N) * np.exp(1j * (2 * np.pi * np.random.rand(N, N) - np.pi )) / np.sqrt(2)) # Preparing steady state: image with correlation + give exponential distribution to the speckle pattern
#image_rand = np.random.rand(N, N) * np.exp(1j*(2*np.pi*np.random.rand(N, N)-np.pi))/np.sqrt(2)

# Create the disk of radius r
Circle = np.ones((N, N))
for x in range(0, N):
    for y in range(0, N):
        if (x - cx) ** 2 + (y - cy) ** 2 > r ** 2:
            Circle[x, y] = 0


# Parameters Fourier shift theorem
X = np.arange(-N/2, N/2)
Y = np.arange(-N/2, N/2)
x, y = np.meshgrid(X, Y)
kx = -1j * 2 * np.pi * x / N    # x-domain (-pi,pi)
ky = -1j * 2 * np.pi * y / N    # y-domain (-pi,pi)
dx = -1                         # Phase shift in x-direction (in mm), dx=1*(1/N) means pixels move 1 spot to RIGHT
dy = 0                          # Phase shift in y-direction (in mm), dy=1*(1/N) means pixels move 1 spot UP


# dx_l = 0 * (1 / N)
# dy_l = 2 * (1 / N)

# Prepare video speckles
db_path = r"\\winbe\owllsci\HFR_blood_pressure\Simulation_tests"
name="test1"
Image_array = []

out = cv2.VideoWriter(db_path + '\\' +  name + '.avi', cv2.VideoWriter_fourcc(*"MJPG"), 35, (image_rand.shape[1], image_rand.shape[0]))

# Speckle dynamics
for t in time:

    # Longitudinal motion blood cells - blood flow velocity
    image_FS = image_rand * np.exp((kx * dx + ky * dy)) #* wave_v(t))

    # Accounting for correlation
    #image_FS = cor * image_FS + (1 - cor) * (np.random.rand(N, N) * np.exp(1j * (2 * np.pi * np.random.rand(N, N) - np.pi)) / np.sqrt(2))

    # Initialization
    image_rand = image_FS

    # Speckle image
    image = np.real(scipy.fft.fft2(image_FS*Circle) * np.conjugate(scipy.fft.fft2(image_FS*Circle)))


    # Lateral motion arteries
    # image = image. * exp(-(kx * dx_l + ky * dy_l) * (0 * 2 * A_v + (A_v / 2) * sin(2 * pi * (f_v * time(t) + phi1_v))));

    # Taking image snapshots
    #imaget0 = fft2(image. * c). * conj(fft2(image. * c))
    #imaget = fft2(image. * c). * conj(fft2(image. * c))

    # Intensity damping due to blood volume - Absorption and scattering
    #image = image * np.exp(-d * mu_eff * wave_hct(t))

    # Create image tensor
    image = image.astype('uint8')
    Image_array.append(image)
    #Image_array = np.array(Image_array)

    # Save frames onto the video

    x = np.repeat(image.astype('uint8'), 3, axis=1)
    x = x.reshape(image.shape[0], image.shape[1], 3)
    out.write(x)

# Release speckle video
out.release()

### ---------------------------------------###

###############################
### Display plots and video ###
###############################

### Show speckle video ###
cap = cv2.VideoCapture(db_path + '\\' +  name + '.avi')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video file")

# Read until video is completed
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        frame = cv2.resize(frame, (600, 400))
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()



# plt.figure()
# for i in range(0,5):
#     plt.imshow(Image_array[i])
#     plt.show()







