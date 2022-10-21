# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:42:59 2022

@author: xavier.mouy
"""

import os
import pandas as pd
import ecosound.core.tools
from ecosound.core.metadata import DeploymentInfo
import ecosound
import tools
import detection
import localization

# #############################################################################
# input parameters ############################################################

in_dir = r"C:\Users\xavier.mouy\Desktop\Darienne\data\3"
out_dir = r"C:\Users\xavier.mouy\Desktop\Darienne\results"
deployment_info_file = r"C:\Users\xavier.mouy\Desktop\Darienne\config_files\deployment_info.csv"  # Deployment metadata

hydrophones_config_file = r"C:\Users\xavier.mouy\Desktop\Darienne\config_files\hydrophones_config_07-HI.csv"  # Hydrophones configuration
detection_config_file = r"C:\Users\xavier.mouy\Desktop\Darienne\config_files\detection_config_large_array.yaml"  # detection parameters
localization_config_file = r"C:\Users\xavier.mouy\Desktop\Darienne\config_files\localization_config_large_array.yaml"  # localization parameters

# #############################################################################
# #############################################################################

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

        # Look up data files for all channels
        audio_files = tools.find_audio_files(in_file, hydrophones_config)

        # run detector on selected channel
        print("Detection in progress...")
        detections = detection.run_detector(
            audio_files["path"][detection_config["AUDIO"]["channel"]],
            audio_files["channel"][detection_config["AUDIO"]["channel"]],
            detection_config,
            deployment_file=deployment_info_file,
        )
        print("-> " + str(len(detections)) + " detections found.")

        # Perform localization using grid search
        print("Localization")
        localizations, PPDs = localization.GridSearch.run_localization(
            audio_files,
            detections,
            tdoa_grid,
            deployment_info_file,
            detection_config,
            hydrophones_config,
            localization_config,
            verbose=False,
        )

        # save results as csv and netcdf file:
        print("Saving results...")
        localizations.to_csv(out_file + ".csv")
        localizations.to_netcdf(out_file + ".nc")
        print(" ")

    else:
        print("File already processed")

print("Processing complete!")
