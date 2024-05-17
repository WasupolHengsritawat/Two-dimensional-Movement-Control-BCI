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

sys.path.append('../Two-dimensional-Movement-Control-BCI')
from mnetools import streams2mnedata, preprocessing

mne.set_log_level(verbose=False)

## Hyperparameters --------------------------------------------------------------------------------------------------------
# Cue Parameters --------------------------------------------------------------
arrow_size = 300             # Arrow Size
n_trials  = [2, 2, 2, 0]  # Number of each class (None, Right, Left, Down)
t_cross = 2                  # Length of cross display [sec]
t_cue = 3                    # Length of arrow display [sec]
t_rest_mean = 2              # Mean of rest session length [sec]
blink_freq = 0               # Arrow blinking frequency [Hz]

# Classification Hyperparameters ----------------------------------------------
# -- |Train Data| --
participant_id = 0
session = 3
initial_run = 1
n_run = 5

# -- |Time Parameters| --
# Offline
tmin= -0.1
tmax= 3
t_baseline_offline = 0.1
# Online
classification_cycle_period = 0.1
classification_window_length = 0.5
t_baseline_online = 0.1

## Cue Preparation --------------------------------------------------------------------------------------------------------
# -- |Create random label sequence| --
labels = np.zeros(n_trials[0], dtype='int')
for i in range(1,len(n_trials)):
    labels = np.concatenate((labels,np.full(n_trials[i],i, dtype='int')))
np.random.shuffle(labels)

# -- |Define Display| --
win = visual.Window(color=(-255, -255, -255), fullscr=True, units = 'pix', screen = 1)

# -- |Shapes| --
def box(pos = (0,0), x = 0, y = 0, color = 'red', size = 1):
    x_neg = x if x < 0 else 0
    x_pos = x if x > 0 else 0

    y_neg = y if y < 0 else 0
    y_pos = y if y > 0 else 0

    return visual.ShapeStim(win, vertices=[(x_neg, -30 + y_neg),(x_pos, -30 + y_neg),(x_pos, 30 + y_pos),(x_neg, 30 + y_pos)], interpolate=True, fillColor=color, pos=pos, size=size)
cross = visual.TextStim(win, text='+', height=50)

# Arrow Vertices
arrow = [[(0,0)],                                                                   # 0 - None
         [(-0.2,0.05),(-0.2,-0.05),(0,-0.05),(0,-0.1),(.2,0),(0,0.1),(0,0.05)],     # 1 - Right
         [(.2,0.05),(.2,-0.05),(0,-0.05),(0,-0.1),(-0.2,0),(0,0.1),(0,0.05)],       # 2 - Left
         [(0.05,0.2),(-0.05,0.2),(-0.05,0),(-0.1,0),(0,-.2),(0.1,0),(0.05,0)]]      # 3 - Down
# Arrow Position
arrow_pos = np.array([(0,0),(-0.3,0),(0.3,0),(0,-0.3)]) # [None, Right, Left, Down]

# Arrow Shapes
arrows = []
for i in range(len(arrow)):
    arrows.append(visual.ShapeStim(win, vertices=arrow[i], interpolate=True, fillColor='red', pos=arrow_pos[i], size=arrow_size))

# Cue State Machine 
def state(s):
    state.value = s
def t_sm():
    t_sm.counter += classification_cycle_period
def t_rest(v):
    t_rest.value = v
def cue_update_flag(v):
    cue_update_flag.value = v

t_sm.counter = 0    # Initial State Machine Time
state(0)            # Initial State

def cue_state_machine(cue):
    if state.value == 0:
        cross.draw()
        t_sm()
        if t_sm.counter > t_cross:
            t_sm.counter = 0
            state(1)
    elif state.value == 1:
        arrows[cue].draw() 
        t_sm()
        if t_sm.counter > t_cue:
            t_sm.counter = 0
            state(2)
            t_rest((t_rest_mean - 0.5) + np.random.rand())
    elif state.value == 2:
        t_sm()
        if t_sm.counter > t_rest.value:
            t_sm.counter = 0
            cue_update_flag(True)
            state(0)

## Model Training Session -------------------------------------------------------------------------------------------------
# -- |Event dictionary| --
# Set up your event name
if session == 1 : events_id = {'right': 0, 'left': 1}
else            : events_id = {'none': 0, 'right': 1, 'left': 2}

epochs_list = []
for i in range(initial_run,initial_run+n_run):
    # -- |File import| --
    streams, header = pyxdf.load_xdf(f"Data/sub-P{participant_id:003d}/ses-S{session:003d}/eeg/sub-P{participant_id:003d}_ses-S{session:003d}_task-Default_run-{i:003d}_eeg.xdf") #Example Data from Lab Recoder

    raw_mne, events = streams2mnedata(streams)
    prepro_mne = preprocessing(raw_mne)
    
    # -- |Epoch Segmentation| --
    epochs = mne.Epochs(prepro_mne, events, tmin= tmin,  tmax= tmax, event_id = events_id, baseline=(tmin,tmin+t_baseline_offline), preload = True,verbose=False,picks = ['C3','Cz','C4','Pz','PO7','PO8','EOG'])

    epochs_list.append(epochs)

epochs = mne.concatenate_epochs(epochs_list)
epochs.set_montage(mne.channels.make_standard_montage('standard_1020'))

# -- |Artifact Removal| --
# Perform regression using the EOG sensor as independent variable and the EEG sensors as dependent variables.
model_plain = EOGRegression(picks="eeg", picks_artifact="eog").fit(epochs)
epochs_clean_plain = model_plain.apply(epochs)

# Redo Baseline Correction
epochs_clean_plain.apply_baseline()

# -- |Model Training| --
CSP_selected = []
CLF_selected = []

events_merge = [[0, 1],[0, 2]]
events_dict_list = [{'none-left': 0, 'left': 2},{'none-right': 0, 'right': 1}]

for i in range(2):
    # Event merged to train different models
    epochs_temp = epochs_clean_plain.copy()
    epochs_temp.events = mne.merge_events(epochs_temp.events, events_merge[i], 0, replace_events=True)
    epochs_temp.event_id = events_dict_list[i]

    # Get EEG data and events
    X = epochs_temp.get_data(copy=False)
    Y = epochs_temp.events[:, -1]

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
    CSP_selected.append(csp_list[ind])
    CLF_selected.append(lr_list[ind])

## Online Session ---------------------------------------------------------------------------------------------------------
# -- |Setup Real Time EEG| --
print("looking for an EEG stream...")
All_streams = resolve_stream()
print(All_streams)

for i in All_streams:
    print(i.name())

Target_LSL = 'obci_eeg1' # Name of Openvibe LSL Streaming
EEG_stream =[stream for stream in All_streams if stream.name() == Target_LSL]
inlet = StreamInlet(EEG_stream[0])

# -- |Time Window Initialization| --
timewindow = np.full((7,500),1e-15)
channels = ['C3','Cz','C4','Pz','PO7','PO8','EOG'] # Set your target EEG channel name
info = mne.create_info(
    ch_names= channels,
    ch_types= ['eeg']*(len(channels) - 1) + ['eog'],
    sfreq= 250,  # OpenBCI Frequency acquistion
    verbose=False
)

# -- |Begin Stimuli| --
cross.draw()
win.flip()
core.wait(3)

timer = 0
j = 0
p = []
while j < len(labels):

    cue_update_flag(False)
    label = labels[j]
    
    # Recieve EEG Data from OpenBCI LSL Streaming
    sample, timestamp = inlet.pull_sample()

    if sample:
        # Update Time Window
        timewindow = np.concatenate([timewindow[:,1:], (np.array([sample[1:]])/1000000).T], axis=1)

        currenttime = time.perf_counter()
        if currenttime >= timer:
            timer = currenttime + classification_cycle_period

            # Display Cues
            cue_state_machine(label)

            # Preprocessing (CAR + Filter)
            timewindow_mne = mne.io.RawArray(timewindow, info, verbose=False)
            timewindow_mne = preprocessing(timewindow_mne)
            
            # Artifact Removal
            timewindow_mne = model_plain.apply(timewindow_mne)

            # Baseline Correction
            realtime_data = timewindow_mne.get_data()[:,-int((classification_window_length)*250):]
            realtime_data = realtime_data - np.array([np.mean(realtime_data[:int((t_baseline_online)*250)], axis = 1)]).T

            # Classification
            p = []
            for i in range(2):
                X_transformed = CSP_selected[i].transform(np.array([realtime_data]))
                p.append(CLF_selected[i].predict_proba(X_transformed)[0,1])

            print(p)

            # -- |Display| --
            # Background Gray Box
            box(pos = (-100,0), x = -360, color='gray').draw()
            box(pos = ( 100,0), x =  360, color='gray').draw()
            # Result Red Box
            box(pos = (-100,0), x = -p[0]*360, color='#751818').draw()
            box(pos = ( 100,0), x =  p[1]*360, color='#751818').draw()
            win.flip()

            if cue_update_flag.value:
                j += 1

win.close()