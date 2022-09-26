import sys
sys.path.append(r"C:\Users\xavier.mouy\Documents\GitHub\ecosound") # Adds higher directory to python modules path.

import numpy as np
import pandas as pd
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
import time
import os
from ecosound.core.audiotools import Sound, upsample
import scipy.signal
from numba import njit
from ecosound.visualization.grapher_builder import GrapherFactory
from ecosound.core.spectrogram import Spectrogram


def defineSphereSurfaceGrid(npoints, radius, origin=[0, 0, 0]):
    # Using the golden spiral method
    # ------------------
    # inputs:
    #   npoints =>  nb of points on the sphere - integer
    #   radius  => radius of the sphere - float
    #   origin  => origin of teh sphere in catesian coordinates- 3 element list
    # ------------------
    # sampling in spherical coordinates
    indices = np.arange(0, npoints, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/npoints)
    theta = np.pi * (1 + 5**0.5) * indices
    # convert to cartesian coordinates
    Sx, Sy, Sz = radius*np.cos(theta) * np.sin(phi), radius*np.sin(theta) * np.sin(phi), radius*np.cos(phi)
    # Adjust origin
    Sx = Sx + origin[0]
    Sy = Sy + origin[1]
    Sz = Sz + origin[2]
    # package in a datafrane
    S = pd.DataFrame({'x': Sx, 'y': Sy, 'z': Sz})
    return S


def defineSphereVolumeGrid(spacing, radius, origin=[0, 0, 0]):
    # ------------------
    # inputs:
    #   spacing =>  distance in meters separatying each receiver - float
    #   radius  => radius of the sphere - float
    #   origin  => origin of the sphere in catesian coordinates- 3 element list
    # ------------------
    # Cube of points (Cartesian coordinates)
    vec = np.arange(-radius, radius+spacing, spacing)
    X, Y, Z = np.meshgrid(vec, vec, vec, indexing='ij')
    Sx = np.reshape(X, X.shape[0]*X.shape[1]*X.shape[2])
    Sy = np.reshape(Y, Y.shape[0]*Y.shape[1]*Y.shape[2])
    Sz = np.reshape(Z, Z.shape[0]*Z.shape[1]*Z.shape[2])
    # Convert to spherical coordinates and remove points with r < radius
    Sr = np.sqrt(Sx**2 + Sy**2 + Sz**2)
    Sr_sphere = Sr <= radius
    Sx_sphere = Sx[Sr_sphere]
    Sy_sphere = Sy[Sr_sphere]
    Sz_sphere = Sz[Sr_sphere]
    # Adjust origin
    Sx_sphere = Sx_sphere + origin[0]
    Sy_sphere = Sy_sphere + origin[1]
    Sz_sphere = Sz_sphere + origin[2]
    # package in a datafrane
    S = pd.DataFrame({'x': Sx_sphere, 'y': Sy_sphere, 'z': Sz_sphere})
    return S


def defineCubeVolumeGrid(x_limits,y_limits,z_limits,spacing):
    # ------------------
    # inputs:
    #   spacing =>  distance in meters separatying each receiver - float
    #   radius  => radius of the sphere - float
    #   origin  => origin of the sphere in catesian coordinates- 3 element list
    # ------------------
    # Cube of points (Cartesian coordinates)
    vec_x = np.arange(min(x_limits), max(x_limits), spacing)
    vec_y = np.arange(min(y_limits), max(y_limits), spacing)
    vec_z = np.arange(min(z_limits), max(z_limits), spacing)
    X, Y, Z = np.meshgrid(vec_x, vec_y, vec_z, indexing='ij')
    Sx = np.reshape(X, X.shape[0]*X.shape[1]*X.shape[2])
    Sy = np.reshape(Y, Y.shape[0]*Y.shape[1]*Y.shape[2])
    Sz = np.reshape(Z, Z.shape[0]*Z.shape[1]*Z.shape[2])
    # package in a datafrane
    S = pd.DataFrame({'x': Sx, 'y': Sy, 'z': Sz})
    return S




def euclidean_dist(df1, df2, cols=['x', 'y', 'z']):
    """
    Calculate euclidean distance between two Pandas dataframes.

    Parameters
    ----------
    df1 : TYPE
        DESCRIPTION.
    df2 : TYPE
        DESCRIPTION.
    cols : TYPE, optional
        DESCRIPTION. The default is ['x','y','z'].

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.linalg.norm(df1[cols].values - df2[cols].values, axis=0)


def calc_hydrophones_distances(hydrophones_coords):
    """
    Calculate Euclidiean distance between each hydrophone of an array.

    Parameters
    ----------
    hydrophones_coords : TYPE
        DESCRIPTION.

    Returns
    -------
    hydrophones_dist_matrix : TYPE
        DESCRIPTION.

    """
    hydrophones_dist_matrix = np.empty((len(hydrophones_coords),len(hydrophones_coords)))
    for index1, row1 in hydrophones_coords.iterrows():
        for index2, row2 in hydrophones_coords.iterrows():
            dist = euclidean_dist(row1, row2)
            hydrophones_dist_matrix[index1, index2] = dist
    return hydrophones_dist_matrix




def find_audio_files(filename, hydrophones_config):
    """ Find corresponding files and channels for all the hydrophones of the array """
    filename = os.path.basename(filename)
    # Define file tail
    for file_root in hydrophones_config['file_name_root']:
        idx = filename.find(file_root)
        if idx >= 0:
            file_tail = filename[len(file_root):]
            break
    # Loop through channels and define all files paths and audio channels
    audio_files = {'path':[], 'channel':[]}
    for row_idx, row_data in hydrophones_config.iterrows():
        file_path = os.path.join(row_data.data_path, row_data.file_name_root + file_tail)
        chan = row_data.audio_file_channel
        audio_files['path'].append(file_path)
        audio_files['channel'].append(chan)
    return audio_files

def stack_waveforms(audio_files, detec, TDOA_max_sec):
    waveform_stack = []
    for audio_file, channel in zip(audio_files['path'], audio_files['channel'] ): # for each channel
        # load waveform
        chan_wav = Sound(audio_file)
        chan_wav.read(channel=channel,
                      chunk=[detec['time_min_offset']-TDOA_max_sec, detec['time_max_offset']+TDOA_max_sec],
                      unit='sec',
                      detrend=True)
        # bandpass filter
        chan_wav.filter('bandpass', [detec['frequency_min'], detec['frequency_max']], verbose=False)
        # stack
        waveform_stack.append(chan_wav.waveform)
    return waveform_stack

def plot_all_channels(audio_files,frame, window_type, nfft, step, fmin, fmax, detections=None, detections_channel=0,verbose=True):
    graph_spectros = GrapherFactory('SoundPlotter', title='Spectrograms', frequency_max=fmax)
    graph_waveforms = GrapherFactory('SoundPlotter', title='Waveforms')
    for audio_file, channel in zip(audio_files['path'], audio_files['channel'] ): # for each channel
        # load waveform
        sound = Sound(audio_file)
        sound.read(channel=channel, unit='sec', detrend=True)
        # Calculates  spectrogram
        spectro = Spectrogram(frame, window_type, nfft, step, sound.waveform_sampling_frequency, unit='sec', verbose=verbose)
        spectro.compute(sound, dB=True, use_dask=False)
        # Crop unused frequencies
        spectro.crop(frequency_min=fmin, frequency_max=fmax, inplace=True)
        # Plot
        graph_spectros.add_data(spectro)
        graph_waveforms.add_data(sound)

    graph_spectros.colormap = 'binary'
    if detections:
        graph_spectros.add_annotation(detections, panel=detections_channel, color='green',label='Detections')
        graph_waveforms.add_annotation(detections, panel=detections_channel, color='green',label='Detections')    

    graph_spectros.show()
    graph_waveforms.show()

def plot_single_channel(audio_file,frame, window_type, nfft, step, fmin, fmax, channel=0,detections=None, verbose=True):

    sound = Sound(audio_file)
    sound.read(channel=channel, unit='sec', detrend=True)
    
    # Calculate spectrogram
    spectro = Spectrogram(frame, window_type, nfft, step, sound.waveform_sampling_frequency, unit='sec', verbose=verbose)
    spectro.compute(sound, dB=True)
    # Crop unused frequencies
    spectro.crop(frequency_min=fmin, frequency_max=fmax, inplace=True)
    
    # Generate plot with waveform and spectrogram
    graph = GrapherFactory('SoundPlotter', title='Recording', frequency_max=1000)
    graph.add_data(sound) # add waveform data
    graph.add_data(spectro) # add spectrogram
    if detections:
        graph.add_annotation(detections, panel=0, color='green', label='Detections') # overlay detections on waveform plot
        graph.add_annotation(detections, panel=1, color='green', label='Detections') # overlay detections on spectrogram plot

    graph.colormap = 'binary'
    graph.show()