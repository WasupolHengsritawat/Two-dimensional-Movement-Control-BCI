#!/usr/bin/env python

import numpy as np
from psychopy import visual, core
from pylsl import StreamInlet, resolve_stream

def box(pos = (0,0), x = 0, y = 0, color = 'red', size = 1):
    x_neg = x if x < 0 else 0
    x_pos = x if x > 0 else 0

    y_neg = y if y < 0 else 0
    y_pos = y if y > 0 else 0

    return visual.ShapeStim(win, vertices=[(-30 + x_neg, -30 + y_neg),(30 + x_pos, -30 + y_neg),(30 + x_pos, 30 + y_pos),(-30 + x_neg, 30 + y_pos)], interpolate=True, fillColor=color, pos=pos, size=size)

# -- |Define Display| --
win = visual.Window(color=(-255, -255, -255), fullscr=True, units = 'pix', screen = 1)

# print("looking for an EEG stream...")
# All_streams = resolve_stream()
# print(All_streams)

# for i in All_streams:
#     print(i.name())

# Target_LSL = 'obci_eeg1' # Name of Openvibe LSL Streaming

# EEG_stream =[stream for stream in All_streams if stream.name() == Target_LSL]

# inlet = StreamInlet(EEG_stream[0])

cross = visual.TextStim(win, text='+', height=50)
cross.draw()
win.flip()
core.wait(3)

while True:
    # # Recieve EEG Data from OpenBCI LSL Streaming
    # sample, timestamp = inlet.pull_sample()
    sample = 1
    if sample:
        cross.draw()
        box(pos = ( 100,0), x =  300, color='grey').draw()
        box(pos = (-100,0), x = -300, color='grey').draw()

        # Preprocessing & Classification
        # Replace random function with classification result of each class

        box(pos = ( 100,0), x =  np.random.randint(0,300), color='red').draw()
        box(pos = (-100,0), x = -np.random.randint(0,100), color='red').draw()
        win.flip()