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

Position = 'Finger_top'

### Display mean square error g2 an NCC ###
folder=r'C:\Users\barbea43\OneDrive - imec\Documents\Pulsatrix\HFR_experiment\std\\' + str(Position)
Files = list(Path(folder).glob('*.npy'))
print(Files)
for file in Files:
    std = np.load(file)
    print(std)