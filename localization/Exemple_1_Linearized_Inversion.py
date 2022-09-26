# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 16:34:48 2022

@author: xavier.mouy
"""
import sys
sys.path.append(r'C:\Users\xavier.mouy\Documents\GitHub\ecosound') # Adds higher directory to python modules path.

import pandas as pd
import ecosound.core.tools
from ecosound.core.metadata import DeploymentInfo
import ecosound
import tools
import detection
import localization


#load configuration files
deployment_info_file = r'.\large-array\deployment_info.csv'
hydrophones_config_file = r'.\large-array\hydrophones_config_07-HI.csv'
detection_config_file = r'.\large-array\detection_config_large_array.yaml'
localization_config_file = r'.\large-array\localization_config_large_array.yaml'

# show deployment file
Deployment = DeploymentInfo()
Deployment.read(deployment_info_file)
print(Deployment.data)

# load and plot hydrophone configuration
hydrophones_config= pd.read_csv(hydrophones_config_file, skipinitialspace=True, dtype={'name': str, 'file_name_root': str}) # load hydrophone coordinates (meters)
localization.plot_localizations3D(hydrophones=hydrophones_config)

# load configuration parameters
detection_config = ecosound.core.tools.read_yaml(detection_config_file)
localization_config = ecosound.core.tools.read_yaml(localization_config_file)

# Look up data files for all channels
infile = r'.\large-array\data\AMAR173.4.20190920T163858Z.wav'
audio_files = tools.find_audio_files(infile, hydrophones_config)

# plot all data 
tools.plot_all_channels(audio_files,
                        detection_config['SPECTROGRAM']['frame_sec'],
                        detection_config['SPECTROGRAM']['window_type'],
                        detection_config['SPECTROGRAM']['nfft_sec'],
                        detection_config['SPECTROGRAM']['step_sec'],
                        detection_config['SPECTROGRAM']['fmin_hz'],
                        detection_config['SPECTROGRAM']['fmax_hz'],
                        detections_channel=detection_config['AUDIO']['channel'],
                        verbose=False)

# run detector on selected channel
print('DETECTION')
detections = detection.run_detector(audio_files['path'][detection_config['AUDIO']['channel']],
                                    audio_files['channel'][detection_config['AUDIO']['channel']],
                                    detection_config,
                                    deployment_file=deployment_info_file)

# Look at detection table
print(detections.data.head())

# plot detections
tools.plot_single_channel(audio_files['path'][detection_config['AUDIO']['channel']],
                          detection_config['SPECTROGRAM']['frame_sec'],
                          detection_config['SPECTROGRAM']['window_type'],
                          detection_config['SPECTROGRAM']['nfft_sec'],
                          detection_config['SPECTROGRAM']['step_sec'],
                          detection_config['SPECTROGRAM']['fmin_hz'],
                          detection_config['SPECTROGRAM']['fmax_hz'],
                          detections=detections,
                          verbose=False)

# Perform localization using linearized inverstion
localizations = localization.LinearizedInversion.run_localization(audio_files, detections, deployment_info_file, detection_config, hydrophones_config, localization_config, verbose=False)

# Show that some detections with no localizations (nan) -> did not converge

# Filter localization results to only keep results with low uncertainty
localizations.filter("x_err_span < 1 & y_err_span < 1 & z_err_span < 1.5", inplace=True)

# plot spectrogram with measurements filtered
tools.plot_single_channel(audio_files['path'][detection_config['AUDIO']['channel']],
                          detection_config['SPECTROGRAM']['frame_sec'],
                          detection_config['SPECTROGRAM']['window_type'],
                          detection_config['SPECTROGRAM']['nfft_sec'],
                          detection_config['SPECTROGRAM']['step_sec'],
                          detection_config['SPECTROGRAM']['fmin_hz'],
                          detection_config['SPECTROGRAM']['fmax_hz'],
                          detections=localizations,
                          verbose=False)

# Plot localizations
localization.plot_localizations3D(localizations=localizations, hydrophones=hydrophones_config)


# Show plot of convergence for a given localization

# save as csv
#localizations.to_csv('large_array_localizations.csv')