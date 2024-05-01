#!/usr/bin/env python

import numpy as np
from psychopy import visual, core
from pylsl import StreamInfo, StreamOutlet

arrow_size = 750             # Arrow Size
n_trials  = [15, 15, 15, 0]  # Number of each class (None, Right, Left, Down)
trial_length = 3             # Length of arrow display [sec]
blink_freq = 0               # Arrow blinking frequency [Hz]

# -- |Create random label sequence| --
labels = np.zeros(n_trials[0], dtype='int')
for i in range(1,len(n_trials)):
    labels = np.concatenate((labels,np.full(n_trials[i],i, dtype='int')))
np.random.shuffle(labels)

# -- |Initialize the stream| --
info = StreamInfo(name='MyMarkerStream', type='Markers', channel_count=1,
                  channel_format='int32', source_id='uniqueid12345')
outlet = StreamOutlet(info)

# -- |Define Arrow Vertices and Arrow Position| --
# Arrow Vertices
arrow = [[(0,0)],                                                                   # 0 - None
         [(-0.2,0.05),(-0.2,-0.05),(0,-0.05),(0,-0.1),(.2,0),(0,0.1),(0,0.05)],     # 1 - Right
         [(.2,0.05),(.2,-0.05),(0,-0.05),(0,-0.1),(-0.2,0),(0,0.1),(0,0.05)],       # 2 - Left
         [(0.05,0.2),(-0.05,0.2),(-0.05,0),(-0.1,0),(0,-.2),(0.1,0),(0.05,0)]]      # 3 - Down
# Arrow Position
arrow_pos = np.array([(0,0),(0.3,0),(-0.3,0),(0,-0.3)]) # [None, Right, Left, Down]

# -- |Define Display| --
win = visual.Window(color=(-255, -255, -255), fullscr=True, units = 'pix', screen = 1)

# -- |Initialize Stimuli Object| --
cross = visual.TextStim(win, text='+', height=50)
arrows = []
for i in range(len(arrow)):
    arrows.append(visual.ShapeStim(win, vertices=arrow[i], interpolate=True, fillColor='red', pos=arrow_pos[i]*arrow_size, size=arrow_size))

# -- |Start an Experiment| --
core.wait(5)    # Participants preparation time

# Start Trials
for trial in range(sum(n_trials)):
    # Cross Display at the begining of each trial (2 sec)
    cross.draw()
    win.flip()
    core.wait(2)

    # Send an event marker before arrow display
    outlet.push_sample(x=[labels[trial]])

    # Arrow Display 
    if blink_freq == 0:
        cross.draw()
        arrows[labels[trial]].draw()
        win.flip()
        core.wait(trial_length)
    else:
        n_blink = trial_length*blink_freq
        for i in range(n_blink):
            cross.draw()
            arrows[labels[trial]].draw()
            win.flip()
            core.wait((2*blink_freq)**-1)

            cross.draw()
            win.flip()
            core.wait((2*blink_freq)**-1)

    # Resting Time (1.5-2.5 sec)
    win.flip()
    core.wait(1.5 + np.random.rand())

win.close()