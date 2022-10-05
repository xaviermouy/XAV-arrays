import ecosound.core
from ecosound.core.audiotools import Sound, upsample
from ecosound.core.spectrogram import Spectrogram
from ecosound.detection.detector_builder import DetectorFactory
import datetime
import os
import platform

def run_detector(infile, channel, config, chunk=None, deployment_file=None):

    sound = Sound(infile)
    # load audio data
    if chunk:
        sound.read(channel=channel, chunk=chunk, unit='sec', detrend=True)
        time_offset_sec = chunk[0]
    else:
        sound.read(channel=channel, detrend=True)
        time_offset_sec = 0

    # Calculates  spectrogram
    spectro = Spectrogram(config['SPECTROGRAM']['frame_sec'],
                                  config['SPECTROGRAM']['window_type'],
                                  config['SPECTROGRAM']['nfft_sec'],
                                  config['SPECTROGRAM']['step_sec'],
                                  sound.waveform_sampling_frequency,
                                  unit='sec',
                                  verbose=False,)
    spectro.compute(sound,
                    config['SPECTROGRAM']['dB'],
                    config['SPECTROGRAM']['use_dask'],
                    config['SPECTROGRAM']['dask_chunks'],)

    spectro.crop(frequency_min=config['SPECTROGRAM']['fmin_hz'],
                         frequency_max=config['SPECTROGRAM']['fmax_hz'],
                         inplace=True,
                         )
    # Denoise
    spectro.denoise(config['DENOISER']['denoiser_name'],
                    window_duration=config['DENOISER']['window_duration_sec'],
                    use_dask=config['DENOISER']['use_dask'],
                    dask_chunks=tuple(config['DENOISER']['dask_chunks']),
                    inplace=True)
    # Detector
    file_timestamp = ecosound.core.tools.filename_to_datetime(sound.file_full_path)[0]
    detector = DetectorFactory(config['DETECTOR']['detector_name'],
                               kernel_duration=config['DETECTOR']['kernel_duration_sec'],
                               kernel_bandwidth=config['DETECTOR']['kernel_bandwidth_hz'],
                               threshold=config['DETECTOR']['threshold'],
                               duration_min=config['DETECTOR']['duration_min_sec'],
                               bandwidth_min=config['DETECTOR']['bandwidth_min_hz']
                               )
    start_time = file_timestamp + datetime.timedelta(seconds=time_offset_sec)
    detections = detector.run(spectro,
                              start_time=start_time,
                              use_dask=config['DETECTOR']['use_dask'],
                              dask_chunks=tuple(config['DETECTOR']['dask_chunks']),
                              debug=False,
                              )
    # add time offset in only a section of recording was analysed.
    detections.data['time_min_offset'] = detections.data['time_min_offset'] + time_offset_sec
    detections.data['time_max_offset'] = detections.data['time_max_offset'] + time_offset_sec

    # add deployment metadata
    detections.insert_metadata(deployment_file, channel=channel)

    # Add file informations
    file_name = os.path.splitext(os.path.basename(sound.file_full_path))[0]
    file_dir = os.path.dirname(sound.file_full_path)
    file_ext = os.path.splitext(sound.file_full_path)[1]
    detections.insert_values(operator_name=platform.uname().node,
                               audio_file_name=file_name,
                               audio_file_dir=file_dir,
                               audio_file_extension=file_ext,
                               audio_file_start_date=ecosound.core.tools.filename_to_datetime(sound.file_full_path)[0]
                               )
    print('...done')

    return detections