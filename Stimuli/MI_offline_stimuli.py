#!/usr/bin/env python

import numpy as np
from psychopy import visual, core
from pylsl import StreamInfo, StreamOutlet

arrow_size = 750          # Arrow Size
n_trials  = [15, 15, 0]   # Number of each class (Right, Left, Down)
trial_length = 3          # Length of arrow display [sec]
blink_freq = 0            # Arrow blinking frequency [Hz]

# -- |Initialize the stream| --
info = StreamInfo(name='MyMarkerStream', type='Markers', channel_count=1,
                  channel_format='int32', source_id='uniqueid12345')
outlet = StreamOutlet(info)

# -- |Define Arrow Vertices and Arrow Position| --
# Arrow Vertices
arrow = [[(-0.2,0.05),(-0.2,-0.05),(0,-0.05),(0,-0.1),(.2,0),(0,0.1),(0,0.05)],     # 0 - Right
         [(.2,0.05),(.2,-0.05),(0,-0.05),(0,-0.1),(-0.2,0),(0,0.1),(0,0.05)],       # 1 - Left
         [(0.05,0.2),(-0.05,0.2),(-0.05,0),(-0.1,0),(0,-.2),(0.1,0),(0.05,0)]]      # 2 - Down
# Arrow Position
arrow_pos = np.array([(0.3,0),(-0.3,0),(0,-0.3)]) # [Right, Left, Down]

# -- |Define Display| --
win = visual.Window(color=(-255, -255, -255), fullscr=True, units = 'pix', screen = 1)

# -- |Initialize Stimuli Object| --
cross = visual.TextStim(win, text='+', height=50)
arrows = []
for i in range(3):
    arrows.append(visual.ShapeStim(win, vertices=arrow[i], interpolate=True, fillColor='red', pos=arrow_pos[i]*arrow_size, size=arrow_size))

# -- |Start an Experiment| --
core.wait(5)    # Participants preparation time

# Start Trials
while sum(n_trials) > 0:
    # Cross Display at the begining of each trial (2 sec)
    cross.draw()
    win.flip()
    core.wait(2)

    # Determine which arrow should be showed up. The each arrow type will be randomly appears until reached the given amount.
    arrow_availables = [i for i, n in enumerate(n_trials) if n > 0]
    ind = np.random.randint(0,len(arrow_availables))
    n_trials[arrow_availables[ind]] -= 1

    # Send an event marker before arrow display
    outlet.push_sample(x=[arrow_availables[ind]])

    # Arrow Display 
    if blink_freq == 0:
        cross.draw()
        arrows[arrow_availables[ind]].draw()
        win.flip()
        core.wait(trial_length)
    else:
        n_blink = trial_length*blink_freq
        for i in range(n_blink):
            cross.draw()
            arrows[arrow_availables[ind]].draw()
            win.flip()
            core.wait((2*blink_freq)**-1)

            cross.draw()
            win.flip()
            core.wait((2*blink_freq)**-1)

    # Resting Time (1.5-2.5 sec)
    win.flip()
    core.wait(1.5 + np.random.rand())

win.close()