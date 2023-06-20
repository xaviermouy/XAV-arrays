# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:46:04 2023

@author: xavier.mouy
"""

import numpy as np

number_freq_bands = 4
min_bandwidth_hz = 300

freq_min = 20
freq_max = 100

freq_band = freq_max - freq_min # bw
if freq_band > min_bandwidth_hz: # if detection bw > 300 Hz
    # break down into sub-bands    
    fb_check = True
    while fb_check == True:
        freq_interval = freq_band / number_freq_bands
        if freq_interval < min_bandwidth_hz:
            fb_check = True
            number_freq_bands = number_freq_bands - 1
        else:
            fb_check = False
    #print(freq_interval)

    freqs = np.arange(freq_min, freq_max,freq_interval)
    freqs = np.append(freqs,freq_max)
    #print(freqs)
    freq_bands=[[freqs[0],freqs[-1]]]
    for idx, val in enumerate(freqs):
        #print(idx)
        if idx < len(freqs)-1:
            freq_bands.append([val, freqs[idx+1]])

else:
    freq_bands=[[freq_min,freq_max]]
print(freq_bands)

#else:
     