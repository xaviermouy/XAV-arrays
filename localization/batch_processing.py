# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:42:59 2022

@author: xavier.mouy
"""

import os
import faulthandler
import pandas as pd
import ecosound.core.tools
from ecosound.core.metadata import DeploymentInfo
from ecosound.core.audiotools import Sound
from ecosound.core.measurement import Measurement
import ecosound
import tools
import detection
import localization
import time
import dask
from dask.distributed import Client, progress
from dask import config as cfg
from datetime import datetime
import shutil
faulthandler.enable()

if __name__ == '__main__':
    #cfg.set({'distributed.scheduler.worker-ttl': None})
    #client = Client(threads_per_worker=1, n_workers=2, memory_limit = '10GB')
    #client = Client(threads_per_worker=1, n_workers=6,memory_limit = '6GB')
    # #############################################################################
    # input parameters ############################################################

    # in_dir = r"/home/xavier/Documents/Darienne_XAV-arrays/data/3"
    # out_dir = r"/home/xavier/Documents/Darienne_XAV-arrays/results"
    # deployment_info_file = r"/home/xavier/Documents/Darienne_XAV-arrays/config_files/deployment_info.csv"  # Deployment metadata
    # hydrophones_config_file = r"/home/xavier/Documents/Darienne_XAV-arrays/config_files/hydrophones_config_07-HI.csv"  # Hydrophones configuration
    # detection_config_file = r"/home/xavier/Documents/Darienne_XAV-arrays/config_files/detection_config_large_array.yaml"  # detection parameters
    # localization_config_file = r"/home/xavier/Documents/Darienne_XAV-arrays/config_files/localization_config_large_array.yaml"  # localization parameters
    # Time_of_day_start = 13
    # Time_of_day_end = 4

    # in_dir = r"/media/xavier/XMOUY_SDD1/Darienne_deployments/Taylor_Islet_LA_2022/AMAR/AMAR173.4.32000.M36-V35-100"
    # out_dir = r"/home/xavier/Documents/Darienne_XAV-arrays/results"
    # deployment_info_file = r"/home/xavier/Documents/Darienne_XAV-arrays/config_files/deployment_info.csv"  # Deployment metadata
    # hydrophones_config_file = r"/home/xavier/Documents/Darienne_XAV-arrays/config_files/hydrophones_config_07-HI.csv"  # Hydrophones configuration
    # detection_config_file = r"/home/xavier/Documents/Darienne_XAV-arrays/config_files/detection_config_large_array.yaml"  # detection parameters
    # localization_config_file = r"/home/xavier/Documents/Darienne_XAV-arrays/config_files/localization_config_large_array.yaml"  # localization parameters
    # Time_of_day_start = 13
    # Time_of_day_end = 4
    # date_deployment = datetime(2022,8,12,23,27)
    # date_retrieval = datetime(2022,8,23,20,37)
    # max_frequency_min = 500

    in_dir = r"/media/xavier/DFO-SSD1/DangerRocksDanvers/AMAR/AMAR173.4.32000.M36-V35-100"
    out_dir = r"/home/xavier/Documents/Darienne_XAV-arrays/Danger_Rock/results"
    deployment_info_file = r"/home/xavier/Documents/Darienne_XAV-arrays/Danger_Rock/config_files/deployment_info_DR.csv"  # Deployment metadata
    hydrophones_config_file = r"/home/xavier/Documents/Darienne_XAV-arrays/Danger_Rock/config_files/hydrophones_config_07-DR.csv"  # Hydrophones configuration
    detection_config_file = r"/home/xavier/Documents/Darienne_XAV-arrays/Danger_Rock/config_files/detection_config_large_array.yaml"  # detection parameters
    localization_config_file = r"/home/xavier/Documents/Darienne_XAV-arrays/Danger_Rock/config_files/localization_config_large_array.yaml"  # localization parameters
    Time_of_day_start = 13
    Time_of_day_end = 4
    date_deployment = datetime(2022,9,8,19,56)
    date_retrieval = datetime(2022,9,16,16,27)
    max_frequency_min = 500


    # #############################################################################
    # #############################################################################

    # create tmp folder
    tmp_dir = os.path.join(out_dir,'tmp')
    if os.path.isdir(tmp_dir) == False:
        os.mkdir(tmp_dir)

    # load deployment metadata
    Deployment = DeploymentInfo()
    Deployment.read(deployment_info_file)

    # load hydrophone configuration
    hydrophones_config = pd.read_csv(
        hydrophones_config_file,
        skipinitialspace=True,
        dtype={"name": str, "file_name_root": str},
    )  # load hydrophone coordinates (meters)

    # load detection parameters
    detection_config = ecosound.core.tools.read_yaml(detection_config_file)

    # load localization parameters
    localization_config = ecosound.core.tools.read_yaml(localization_config_file)

    # create, save, and load 3D grid of TDOAs
    tdoa_grid_file = os.path.join(out_dir, "tdoa_grid.npz")
    localization.GridSearch.create_tdoa_grid(
        localization_config["GRIDSEARCH"]["x_limits_m"],
        localization_config["GRIDSEARCH"]["y_limits_m"],
        localization_config["GRIDSEARCH"]["z_limits_m"],
        localization_config["GRIDSEARCH"]["spacing_m"],
        hydrophones_config,
        localization_config["TDOA"]["ref_channel"],
        localization_config["ENVIRONMENT"]["sound_speed_mps"],
        tdoa_grid_file,
    )
    tdoa_grid = localization.GridSearch.load_tdoa_grid(tdoa_grid_file)

    # find all ausio files to process in the in_dir folder
    files = ecosound.core.tools.list_files(
        in_dir,
        ".wav",
        recursive=False,
        case_sensitive=True,
    )

    # Process each audio file
    nfiles = len(files)
    for idx, in_file in enumerate(files):
        print(idx + 1, "/", nfiles, os.path.split(in_file)[1])
        out_file = os.path.join(out_dir, os.path.split(in_file)[1])
        if os.path.exists(out_file + ".nc") is False:
            file_datetime = ecosound.core.tools.filename_to_datetime(out_file)[0]
            file_tod = file_datetime.hour
            #only process files when intrument is in water
            if (file_datetime > date_deployment) and (file_datetime < date_retrieval):
                # only process if camera is ON (during day time)
                if (file_tod > Time_of_day_start) or (file_tod < Time_of_day_end):
                    try:
                        # create file folder in the tmp dir
                        tmp_dir_file = os.path.join(tmp_dir, os.path.split(in_file)[1])
                        if os.path.isdir(tmp_dir_file) == False:
                            os.mkdir(tmp_dir_file)

                        # Look up data files for all channels
                        audio_files = tools.find_audio_files(in_file, hydrophones_config)

                        # run detector on selected channel
                        print("Detection in progress...")
                        tic = time.perf_counter()
                        detections = detection.run_detector(
                            audio_files["path"][detection_config["AUDIO"]["channel"]],
                            audio_files["channel"][detection_config["AUDIO"]["channel"]],
                            detection_config,
                            deployment_file=deployment_info_file,
                        )
                        toc = time.perf_counter()
                        print(f"Elapsed time: {toc - tic:0.4f} seconds")

                        # remove detections within 0.5 s from borders to avoid issues
                        chan_wav = Sound(in_file)
                        detections.filter('time_min_offset > 0.5', inplace=True)
                        detections.filter('time_max_offset <'+ str(chan_wav.file_duration_sec-0.5), inplace=True)                        
                        #detections.data.reset_index(drop=True,inplace=True)
                        print("-> " + str(len(detections)) + " detections found.")
                        
                        # Removes high freq detections
                        #detections.filter('frequency_min <'+ str(max_frequency_min), inplace=True)
                        detections.filter('frequency_min < 500', inplace=True)

                        # remove detection with small bandwidth (< 100 Hz)
                        bw = detections.data['frequency_max'] - detections.data['frequency_min']
                        detections.data = detections.data[bw>50]
                        detections.data.reset_index(drop=True,inplace=True)                        
                        print("-> " + str(len(detections)) + " detections (after filtering).")

                        # Perform localization using grid search
                        print("Localization")
                        tic2 = time.perf_counter()

                        print(tmp_dir_file)
                        localizations = localization.GridSearch.run_localization(
                            audio_files,
                            detections,
                            tdoa_grid,
                            deployment_info_file,
                            detection_config,
                            hydrophones_config,
                            localization_config,
                            tmp_dir_file,
                            verbose=False,
                        )
                        toc2 = time.perf_counter()
                        print(f"Elapsed time: {toc2 - tic2:0.4f} seconds")

                        # save results as csv and netcdf file:
                        print("Saving results...")
                        localizations.to_netcdf(out_file)
                        localizations.to_raven(out_dir)
                        print(" ")

                        # delete tmp folder and files
                        #os.rmdir(tmp_dir_file)
                        shutil.rmtree(tmp_dir_file)
                    except BaseException as e: 
                        print(e)
                        print('Processing failed...')
                else:
                    print("Time of the file outside of the analysis effort. File not processed.")
            else:
                print("File outside of deployment time. File not processed.")
        else:
            print("File already processed")

    print("Processing complete!")
