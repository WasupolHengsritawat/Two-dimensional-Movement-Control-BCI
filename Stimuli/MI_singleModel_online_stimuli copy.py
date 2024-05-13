#!/usr/bin/env python

import time
import numpy as np
from psychopy import visual, core
from pylsl import StreamInlet, resolve_stream

import pyxdf
import mne
from mne.decoding import CSP
from mne.preprocessing import EOGRegression
from mne.stats import permutation_cluster_1samp_test as pcluster_test
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split

import sys
 
# setting path
sys.path.append('../Two-dimensional-Movement-Control-BCI')
 
# importing
from mnetools import streams2mnedata, preprocessing

mne.set_log_level(verbose=False)

## Model Training Session -------------------------------------------------------------------------------------------------
# -- |Data details| --
participant_id = 0
session = 3
# -- |Data Selection| --
initial_run = 1
n_run = 5

# -- |Parameters| --
tmin= -0.5
tmax= 3
classification_cycle_period = 0.2
classification_window_length = 0.5

# -- |Event dictionary| --
# Set up your event name
if session == 1 : events_id = {'right': 0, 'left': 1}
else            : events_id = {'none': 0, 'right': 1, 'left': 2}

# -- |Local parameters|--
epochs_list = [] 

for i in range(initial_run,initial_run+n_run):
    # -- |File import| --
    streams, header = pyxdf.load_xdf(f"Data/sub-P{participant_id:003d}/ses-S{session:003d}/eeg/sub-P{participant_id:003d}_ses-S{session:003d}_task-Default_run-{i:003d}_eeg.xdf") #Example Data from Lab Recoder

    raw_mne, events = streams2mnedata(streams)
    prepro_mne = preprocessing(raw_mne)
    
    # -- |Epoch Segmentation| --
    epochs = mne.Epochs(prepro_mne, events, tmin= tmin,  tmax= tmax, event_id = events_id, preload = True,verbose=False,picks = ['C3','Cz','C4','Pz','PO7','PO8','EOG'])

    epochs_list.append(epochs)

epochs = mne.concatenate_epochs(epochs_list)
epochs.set_montage(mne.channels.make_standard_montage('standard_1020'))

# -- |Artifact Removal| --
# Perform regression using the EOG sensor as independent variable and the EEG
# sensors as dependent variables.
model_plain = EOGRegression(picks="eeg", picks_artifact="eog").fit(epochs)

epochs_clean_plain = model_plain.apply(epochs)
# Redo Baseline Correction
epochs_clean_plain.apply_baseline()

# -- |Model Training| --
# Get EEG data and events
X = epochs_clean_plain[['right','left']].get_data(copy=False)
Y = epochs_clean_plain[['right','left']].events[:, -1]

csp_list = []
lr_list = []
acc_list = []

for i in range(2, len(epochs.ch_names) + 1):
    # -- |Features Extraction| --
    # Initilize CSP
    csp = CSP(n_components = i, norm_trace = False)

    # Fit CSP to data 
    csp.fit(X,Y)
    csp_list.append(csp)

    # Transform data into CSP space
    X_transformed = csp.transform(X)

    # -- |Classification| --
    # Split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_transformed, Y, test_size = 0.2, random_state = 42, stratify=Y)

    # Classification 
    lr = Pipeline([('LR', LogisticRegression())])
    lr.fit(X_train, Y_train)
    lr_list.append(lr)

    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    acc_list.append(accuracy)

# -- |Select CSP and models which gives maximum accuracy| --
ind = np.argmax(acc_list) 
CSP_selected = csp_list[ind]
CLF_selected = lr_list[ind]

## Online Session ---------------------------------------------------------------------------------------------------------
def box(pos = (0,0), x = 0, y = 0, color = 'red', size = 1):
    x_neg = x if x < 0 else 0
    x_pos = x if x > 0 else 0

    y_neg = y if y < 0 else 0
    y_pos = y if y > 0 else 0

    return visual.ShapeStim(win, vertices=[(x_neg, -30 + y_neg),(x_pos, -30 + y_neg),(x_pos, 30 + y_pos),(x_neg, 30 + y_pos)], interpolate=True, fillColor=color, pos=pos, size=size)

# -- |Define Display| --
win = visual.Window(color=(-255, -255, -255), fullscr=True, units = 'pix', screen = 1)

print("looking for an EEG stream...")
All_streams = resolve_stream()
print(All_streams)

for i in All_streams:
    print(i.name())

Target_LSL = 'obci_eeg1' # Name of Openvibe LSL Streaming

EEG_stream =[stream for stream in All_streams if stream.name() == Target_LSL]

inlet = StreamInlet(EEG_stream[0])

cross = visual.TextStim(win, text='+', height=50)
cross.draw()
win.flip()
core.wait(3)

# Create Time Window
timewindow = np.full((7,500),1e-15)
channels = ['C3','Cz','C4','Pz','PO7','PO8','EOG'] # Set your target EEG channel name
info = mne.create_info(
    ch_names= channels,
    ch_types= ['eeg']*(len(channels) - 1) + ['eog'],
    sfreq= 250,  # OpenBCI Frequency acquistion
    verbose=False
)

timer = 0
p = [0,0]
while True:
    # Recieve EEG Data from OpenBCI LSL Streaming
    sample, timestamp = inlet.pull_sample()
    
    # Update Time Window
    timewindow = np.concatenate([timewindow[:,1:], (np.array([sample[1:]])/1000000).T], axis=1)

    if sample:
        currenttime = time.perf_counter()
        if currenttime >= timer:
            timer = currenttime + classification_cycle_period

            # Preprocessing & Classification
            # Replace random function with classification result of each class
            timewindow_mne = mne.io.RawArray(timewindow, info, verbose=False)
            timewindow_mne = preprocessing(timewindow_mne)
            timewindow_mne = model_plain.apply(timewindow_mne)
            
            realtime_data = timewindow_mne.get_data()[:,-int(classification_window_length*250):]
            realtime_data = realtime_data - np.array([np.mean(realtime_data[:int((0-tmin)*250)], axis = 1)]).T

            X_transformed = CSP_selected.transform(np.array([realtime_data]))
            p = CLF_selected.predict_proba(X_transformed)[0]
            
            print(p)

    # -- |Display| --
    cross.draw()
    # Background Gray Box
    box(pos = (-100,0), x = -360, color='gray').draw()
    box(pos = ( 100,0), x =  360, color='gray').draw()
    # Result Red Box
    box(pos = (-100,0), x = -p[1]*360, color='red').draw()
    box(pos = ( 100,0), x =  p[0]*360, color='red').draw()
    win.flip()