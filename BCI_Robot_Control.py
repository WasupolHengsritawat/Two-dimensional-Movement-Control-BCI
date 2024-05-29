import rtde_control
import rtde_receive
import math
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

IP_address = "169.254.61.32"

# --|Connect to the robot|--
rtde_c = rtde_control.RTDEControlInterface(IP_address) 
rtde_r = rtde_receive.RTDEReceiveInterface(IP_address)  

## Hyperparameters --------------------------------------------------------------------------------------------------------
# Robot Parameters ------------------------------------------------------------
rest_period_length = 1
joint_list = [0,4,5]

# Cue Parameters --------------------------------------------------------------
screen = 0
bar_length = 400             # Bar length
t_cue = 3                    # Length of classification period [sec]

# Motor Imagery Classification Hyperparameters --------------------------------
# -- |Train Data| --
participant_id = 4
session_S = 3
initial_run_S = 1
n_run_S = 5

# -- |Decision Threshold| --
decision_threshold = 0       # Difference between probability of classes. This will allow model to predict 'No Imagery' class

# -- |Time Parameters| --
# Offline
tmin_S = -0.5
tmax_S = 3
t_baseline_offline_S = 0.5
# Online
timewindow_lenght = 3
classification_cycle_period = 0.2
classification_window_length = 0.5
t_baseline_online_S = 0.2

# Blinking Detection  Hyperparameters -----------------------------------------
# -- |Train Data| --
session_B = 3
initial_run_B = 1
n_run_B = 2

# -- |Time Parameters| --
# Offline
tmin_B = -0.5
tmax_B = 3
t_baseline_offline_B = 0.5
# Online
t_baseline_online_B = 0.5

## Cue Preparation --------------------------------------------------------------------------------------------------------
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

# -- |Number of Boxes| --
init_pos = 150
nbox = int(t_cue//classification_cycle_period)
box_length = int((bar_length*0.8)//nbox)
box_space  = int(bar_length//nbox)

## Robot Control -------------------------------------------------------------------------------------------------------------
def d2r(degrees):
    return degrees * (math.pi / 180)

def state(s):
    state.value = s
def t_rest(t):
    t_rest.value = t
def dir(d):
    dir.value = d
def speed(sp):
    speed.value = sp

# -- |Variable Initialization| --
state(0)
t_rest(0)
dir(0)
speed([0,0,0,0,0,0])

def velo_joint(joint, direction):
    t_start = rtde_c.initPeriod()
    speed.value[joint] = 1.5 * direction
    rtde_c.speedJ(speed.value, 0.01, 1.0/500)
    if rtde_r.getActualQ()[joint] < d2r(-100) and direction == -1:
        dir(1)
        rtde_c.speedStop()
        print("Change Direction 1")
    if rtde_r.getActualQ()[joint] > d2r(100) and direction == 1:
        dir(-1)
        rtde_c.speedStop()
        print("Change Direction -1")
    rtde_c.waitPeriod(t_start)

def FSM(current_joint, blink, MI_classification_result):
    if state.value == 0:
        if blink == 1:
            state(1)
    elif state.value == 1:
        if MI_classification_result != 0:
            dir(MI_classification_result)
            state(2)
    elif state.value == 2:
        velo_joint(current_joint, dir.value)

        if blink == 1:
            t_rest(time.perf_counter() + rest_period_length)
            speed([0,0,0,0,0,0])
            rtde_c.speedStop()
            state(3)
    elif state.value == 3:
        if time.perf_counter() > t_rest.value:
            state(0)

## MI Model Training Session -------------------------------------------------------------------------------------------------
# -- |Event dictionary| --
# Set up your event name
if session_S == 1 : events_id_S = {'right': 0, 'left': 1}
else              : events_id_S = {'none': 0, 'right': 1, 'left': 2}

epochs_list = [] 
for i in range(initial_run_S,initial_run_S+n_run_S):
    # -- |File import| --
    streams, header = pyxdf.load_xdf(f"Data/sub-P{participant_id:003d}/ses-S{session_S:003d}/eeg/sub-P{participant_id:003d}_ses-S{session_S:003d}_task-Default_run-{i:003d}_eeg.xdf") #Example Data from Lab Recoder

    raw_mne, events = streams2mnedata(streams)
    prepro_mne = preprocessing(raw_mne)
    
    # -- |Epoch Segmentation| --
    epochs = mne.Epochs(prepro_mne, events, tmin= tmin_S,  tmax= tmax_S, event_id = events_id_S, baseline=(tmin_S,tmin_S+t_baseline_offline_S), preload = True,verbose=False,picks = ['C3','Cz','C4','Pz','PO7','PO8','EOG'])

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
CSP_selected = csp_list[ind]
CLF_selected = lr_list[ind]

## Blinking Threshold Tuning ----------------------------------------------------------------------------------------------
# -- |Event dictionary| --
# Set up your event name
events_id_B = {'none': 0, 'singleBlink': 1, 'doubleBlink': 2}

# -- |Local parameters|--
epochs_list = [] 

for i in range(initial_run_B,initial_run_B+n_run_B):
    # -- |File import| --
    streams, header = pyxdf.load_xdf(f"Data/sub-P{participant_id:003d}/ses-B{session_B:003d}/eeg/sub-P{participant_id:003d}_ses-B{session_B:003d}_task-Default_run-{i:003d}_eeg.xdf") #Example Data from Lab Recoder

    raw_mne, events = streams2mnedata(streams)
    # -- |Common Average Reference| --
    mne_car = raw_mne.copy().set_eeg_reference('average', verbose=False)

    # -- |Bandpass filter| --
    mne_filtered = mne_car.filter(l_freq=2.0, h_freq=15.0, fir_design='firwin',picks ='all', verbose=False)
    
    # -- |Epoch Segmentation| --
    epochs = mne.Epochs(mne_filtered, events, tmin= tmin_B,  tmax= tmax_B, event_id = events_id_B, baseline=(tmin_B,tmin_B+t_baseline_offline_B), preload = True,verbose=False,picks = ['C3','Cz','C4','PO7','Pz','PO8','EOG'])

    epochs_list.append(epochs)

epochs_blink = mne.concatenate_epochs(epochs_list)
epochs_blink.set_montage(mne.channels.make_standard_montage('standard_1020'))

epochs_blink.events = mne.merge_events(epochs_blink.events, [0, 1], 0, replace_events=True)
epochs_blink.events = mne.merge_events(epochs_blink.events, [2], 1, replace_events=True)
epochs_blink.event_id = {'none': 0, 'doubleBlink': 1}

X_B = epochs_blink.get_data(copy=False, picks = 'EOG')
Y_B = epochs_blink.events[:, -1]

# -- |Optimize Threshold| --
acc = []
A = []
for a in np.array(range(1,300))/100:
    threshold = np.mean(X_B) + a*np.std(X_B)
    Y_pred = []
    for i in range(X_B.shape[0]):
        if sum(X_B[i,0,:] > threshold) >= 2: Y_pred.append(1)
        else: Y_pred.append(0)
    Y_pred = np.array(Y_pred)

    A.append(a)
    acc.append(accuracy_score(Y_B, Y_pred))

a = A[np.argmax(acc)]
threshold = np.mean(X_B) + a*np.std(X_B)

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
timewindow = np.full((7,int(250*timewindow_lenght)),1e-15)
channels = ['C3','Cz','C4','Pz','PO7','PO8','EOG'] # Set your target EEG channel name
info = mne.create_info(
    ch_names= channels,
    ch_types= ['eeg']*(len(channels) - 1) + ['eog'],
    sfreq= 250,  # OpenBCI Frequency acquistion
    verbose=False
)

# -- |Begin Stimuli| --
cross = visual.TextStim(win, text='+', height=50)
cross.draw()
win.flip()
core.wait(3)

timer = 0
j = 0
p = [0,0]
Pscores = np.array([0,0])
blink = 0
MI_classification_result = 0

bootup_time = time.perf_counter() + t_cue + 1

last_state = 0

while j < len(joint_list):

    current_joint = joint_list[j]

    # Recieve EEG Data from OpenBCI LSL Streaming
    sample, timestamp = inlet.pull_sample()

    if time.perf_counter() > bootup_time:
        FSM(current_joint, blink, MI_classification_result)
    print('state =', state.value,' Blink =', blink)

    if sample:
        # Update Time Window
        timewindow = np.concatenate([timewindow[:,1:], (np.array([sample[1:]])/1000000).T], axis=1)

        currenttime = time.perf_counter()
        if currenttime >= timer:
            timer = currenttime + classification_cycle_period

            # -- |MI Online Classification| --
            # Preprocessing (CAR + Filter)
            timewindow_mne = mne.io.RawArray(timewindow, info, verbose=False)
            timewindow_mne_S = preprocessing(timewindow_mne)

            # Artifact Removal
            timewindow_mne_S = model_plain.apply(timewindow_mne_S)
            
            # Baseline Correction
            realtime_data_S = timewindow_mne_S.get_data()[:,-int(classification_window_length*250):]
            realtime_data_S = realtime_data_S - np.array([np.mean(realtime_data_S[:int((t_baseline_online_S)*250)], axis = 1)]).T

            # Classification
            X_transformed = CSP_selected.transform(np.array([realtime_data_S]))
            p = CLF_selected.predict_proba(X_transformed)[0]

            if state.value == 1:
                if np.abs(p[0] - p[1]) > decision_threshold:
                    Pscores[np.argmax(p)] += 1

            # print(p)
            print(Pscores)
            
            if np.max(Pscores) >= nbox:
                if np.argmax(Pscores) == 0:
                    MI_classification_result = -1
                else:
                    MI_classification_result = 1

            # -- |Online Blink Detection| -- 
            # Preprocessing (CAR + Filter)
            timewindow_mne_B = timewindow_mne.copy().set_eeg_reference('average', verbose=False)
            timewindow_mne_B = timewindow_mne_B.filter(l_freq=2.0, h_freq=15.0, fir_design='firwin',picks ='all', verbose=False)

            # Baseline Correction
            realtime_data_B = timewindow_mne_B.get_data()
            realtime_data_B = realtime_data_B - np.array([np.mean(realtime_data_B[:int((t_baseline_online_B)*250)], axis = 1)]).T

            if time.perf_counter() > bootup_time and (state.value == 0 or state.value == 2):
                if sum(realtime_data_B[-1] > threshold) >= 2: blink = 1

            # -- |Display| --
            cross.draw()
            # Results Bar
            for k in range(nbox):
                colors = np.where(np.array([k,k]) < Pscores,'#751818','gray')
                box(pos = (-(init_pos + k*box_space),0), x = -box_length, color=colors[0]).draw()
                box(pos = ( (init_pos + k*box_space),0), x =  box_length, color=colors[1]).draw()
            win.flip()

            if state.value == 1:
                blink = 0

            if state.value == 3:
                Pscores = np.array([0,0])
                blink = 0
                MI_classification_result = 0

            if state.value == 0 and last_state == 3:
                j += 1

            last_state = state.value

win.close()




