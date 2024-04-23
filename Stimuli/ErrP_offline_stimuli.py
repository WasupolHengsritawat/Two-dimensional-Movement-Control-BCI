#!/usr/bin/env python

import numpy as np
from psychopy import visual, core
from pylsl import StreamInfo, StreamOutlet

n_trials  = [40, 10]   # Number of each class (Correct, Incorrect)

label = []
for i in range(n_trials[0]):
    label.append(1)
for  i in range(n_trials[1]):
    label.append(-1)

np.random.shuffle(label)

def box(pos, color, size = 1):
    return visual.ShapeStim(win, vertices=[(-20,-20),(20,-20),(20,20),(-20,20)], interpolate=True, fillColor=color, pos=pos, size=size)

# -- |Initialize the stream| --
info = StreamInfo(name='MyMarkerStream', type='Markers', channel_count=1,
                  channel_format='int32', source_id='uniqueid12345')
outlet = StreamOutlet(info)

# -- |Define Display| --
win = visual.Window(color=(-255, -255, -255), fullscr=True, units = 'pix', screen = 1)

cross = visual.TextStim(win, text='+', height=50)
cross.draw()
win.flip()
core.wait(3)

video_refresh_rate = 120
trial_lenght = 1.5
init_pos = (0,0)

for trial in range(sum(n_trials)):
    correctness = label[trial]
    dir = 1 if np.random.randint(0,2) == 1 else -1
    color = 'red' if dir == 1 else 'blue'
    step_zize = 0.75 + np.random.rand()

    if max(np.abs(init_pos[0] + dir*int(video_refresh_rate*trial_lenght)*step_zize), np.abs(init_pos[0] + dir*correctness*int(video_refresh_rate*trial_lenght)*step_zize)) > (win.size[0] - 40)/2:
        init_pos = (0,0)

    box((init_pos[0] + dir*int(video_refresh_rate*trial_lenght)*step_zize,init_pos[1]), color).draw()
    box((init_pos[0],init_pos[1]), 'green').draw()
    cross.draw()
    win.flip()
    core.wait(1)

    outlet.push_sample(x=[correctness])

    if correctness == -1:
        correctness *= (0.25 + np.random.rand()) # to make step size mismatched with correct step size

    # -- |Cursor Run| --
    for i in range(int(video_refresh_rate*trial_lenght)):
        box((init_pos[0] + dir*video_refresh_rate*trial_lenght*step_zize,init_pos[1]), color).draw()
        box((init_pos[0] + correctness*dir*i*step_zize,init_pos[1]), 'green').draw()
        cross.draw()
        win.flip()
        core.wait(1/video_refresh_rate)

    init_pos = (init_pos[0] + correctness*dir*int(video_refresh_rate*trial_lenght)*step_zize,init_pos[1])

win.close()