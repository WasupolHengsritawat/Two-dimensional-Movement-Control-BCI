#!/usr/bin/env python

import time
import numpy as np
from psychopy import visual, core
from pylsl import StreamInlet, resolve_stream

import pyxdf
import mne
from mne.decoding import CSP
from mne.preprocessing import EOGRegression
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys

sys.path.append('../Two-dimensional-Movement-Control-BCI')
from mnetools import streams2mnedata, preprocessing

mne.set_log_level(verbose=False)

## Hyperparameters --------------------------------------------------------------------------------------------------------
# Cue Parameters --------------------------------------------------------------
screen = 0
arrow_size = 300             # Arrow Size
bar_length = 400             # Bar length
n_trials  = [15, 15, 15, 0]     # Number of each class (None, Right, Left, Down)
t_cross = 2                  # Length of cross display [sec]
t_cue = 3                    # Length of arrow display [sec]
t_rest_mean = 2              # Mean of rest session length [sec]
blink_freq = 0               # Arrow blinking frequency [Hz]

# Classification Hyperparameters ----------------------------------------------
# -- |Train Data| --
participant_id = 4
session = 2
initial_run = 1
n_run = 5

# -- |Decision Threshold| --
decision_threshold = [0.5,0.5] # Left Right

# -- |Time Parameters| --
# Offline
tmin= -0.5
tmax= 3
t_baseline_offline = 0.5
# Online
timewindow_lenght = 3
classification_cycle_period = 0.2
classification_window_length = 0.4
t_baseline_online = 0.1

## Cue Preparation --------------------------------------------------------------------------------------------------------
# -- |Create random label sequence| --
labels = np.zeros(n_trials[0], dtype='int')
for i in range(1,len(n_trials)):
    labels = np.concatenate((labels,np.full(n_trials[i],i, dtype='int')))
np.random.shuffle(labels)

# -- |Define Display| --
win = visual.Window(color=(-255, -255, -255), fullscr=True, units = 'pix', screen = screen)

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

# -- |Cue State Machine| --
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
        cross.draw()
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

# -- |Number of Boxes| --
init_pos = 150
nbox = int(t_cue//classification_cycle_period)
box_length = int((bar_length*0.8)//nbox)
box_space  = int(bar_length//nbox)

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

        # -- |Classification| --
        # Split data into training and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify=Y)

        # Fit CSP to data 
        csp.fit(X_train,Y_train)
        csp_list.append(csp)

        # Transform data into CSP space
        X_train_transformed = csp.transform(X_train)
        X_test_transformed = csp.transform(X_test)

        # Classification 
        lr = Pipeline([('LR', LogisticRegression())])
        lr.fit(X_train_transformed, Y_train)
        lr_list.append(lr)

        y_pred = lr.predict(X_test_transformed)
        accuracy = accuracy_score(Y_test, y_pred)
        acc_list.append(accuracy)

    # -- |Select CSP and models which gives maximum accuracy| --
    ind = np.argmax(acc_list) 
    # ind = len(acc_list) - 1
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
timewindow = np.full((7,int(timewindow_lenght*250)),1e-15)
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
Pscores = np.array([0,0])
Nscores = np.array([0,0])
Pscores_history = []
Nscores_history = []
while j < len(labels):

    cue_update_flag(False)
    label = labels[j]
    
    # Recieve EEG Data from OpenBCI LSL Streaming
    sample, timestamp = inlet.pull_sample()

    if sample:
        # Update Time Window
        timewindow = np.concatenate([timewindow[:,1:], (np.array([sample[1:]])).T/1000000], axis=1)

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
                prob = CLF_selected[i].predict_proba(X_transformed)[0,1]

                if prob > decision_threshold[i] and state.value == 1: Pscores[i] += 1
                elif prob <= decision_threshold[i] and state.value == 1: Nscores[i] += 1
                p.append(prob)

            print(p)

            # -- |Display| --
            # Results Bar
            for k in range(nbox):
                colors = np.where(np.array([k,k]) < Pscores,'#751818','gray')
                box(pos = (-(init_pos + k*box_space),0), x = -box_length, color=colors[0]).draw()
                box(pos = ( (init_pos + k*box_space),0), x =  box_length, color=colors[1]).draw()
            win.flip()

            if cue_update_flag.value:
                Pscores_history.append([label, Pscores])
                Nscores_history.append([label, Nscores])
                Pscores = np.array([0,0])
                Nscores = np.array([0,0])
                j += 1

win.close()

## Performance Evaluation -------------------------------------------------------------------------------------------------
def confusion_matrix(TP,FN,FP,TN):
    '''Display Confusion Matrix'''
    # -- |Results calculation| --
    accuracy = (TP + TN)/(TP + FP + TN + FN)

    if TP + FP == 0:
        precision = np.NaN
    else:
        precision = TP/(TP + FP)
    
    if TP + FN == 0:
        recall = np.NaN
    else:
        recall = TP/(TP + FN)
    
    if precision + recall == 0:
        f1 = np.NaN
    else:
        f1 = 2*(precision*recall)/(precision + recall)

    # -- |Results display| --
    print('Confusion Matrix')
    print('--------------------------------------')
    print('|            |       Predicted       |')
    print('|            -------------------------   Accuracy  = {0:}'.format(accuracy))
    print('|            |     P     |     N     |')
    print('--------------------------------------   Precision = {0:}'.format(precision))
    print("|        | P |   {0:^5d}   |   {1:^5d}   |   Recall    = {2:}".format(TP, FN, recall))
    print('| Actual -----------------------------')
    print("|        | N |   {0:^5d}   |   {1:^5d}   |   F1-score  = {2:}".format(FP, TN, f1))
    print('--------------------------------------')

    return [accuracy, precision, recall, f1]

positive = []
negative = []
for l in range(np.max(list(events_id.values())) + 1):
    Pscores_total = np.array([sh[1] for sh in Pscores_history if sh[0] == l])
    Pscores_total = np.sum(Pscores_total, axis = 0)
    if np.sum(Pscores_total) == 0:
        Pscores_total = np.array([0,0])
    positive.append(Pscores_total)

    Nscores_total = np.array([sh[1] for sh in Nscores_history if sh[0] == l])
    Nscores_total = np.sum(Nscores_total, axis = 0)
    if np.sum(Nscores_total) == 0:
        Nscores_total = np.array([0,0])
    negative.append(Nscores_total)

positive = np.array(positive)
negative = np.array(negative)
print('Left Model Performance')
confusion_matrix(positive[events_id['left']][0],negative[events_id['left']][0],(np.sum(positive[:,0]) - positive[events_id['left']])[0],(np.sum(negative[:,0]) - negative[events_id['left']])[0])
print('Right Model Performance')
confusion_matrix(positive[events_id['right']][1],negative[events_id['right']][1],(np.sum(positive[:,1]) - positive[events_id['right']])[1],(np.sum(negative[:,1]) - negative[events_id['right']])[1])