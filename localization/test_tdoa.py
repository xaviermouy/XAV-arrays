# -*- coding: utf-8 -*-
"""
Created on Thu May 25 13:50:39 2023

@author: xavier.mouy
"""
import os
import pandas as pd
import ecosound.core.tools
from ecosound.core.metadata import DeploymentInfo
from ecosound.core.audiotools import Sound
import ecosound
import tools
import detection
import localization
from ecosound.core.measurement import Measurement


in_dir = r"C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\Darienne_data\Taylor-Islet_LA_dep2\data\1"
out_dir = r"C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\Darienne_data\Taylor-Islet_LA_dep2\results"
deployment_info_file = r"C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\Darienne_data\Taylor-Islet_LA_dep2\config_files\deployment_info.csv"  # Deployment metadata
hydrophones_config_file = r"C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\Darienne_data\Taylor-Islet_LA_dep2\config_files\hydrophones_config_07-HI.csv"  # Hydrophones configuration
detection_config_file = r"C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\Darienne_data\Taylor-Islet_LA_dep2\config_files\detection_config_large_array.yaml"  # detection parameters
localization_config_file = r"C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\Darienne_data\Taylor-Islet_LA_dep2\config_files\localization_config_large_array.yaml"  # localization parameters


detec_file = r'C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\Darienne_data\Taylor-Islet_LA_dep2\results\New folder/AMAR173.1.20220821T180710Z.wav.nc'

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

# detections
detections = Measurement()
detections.from_netcdf(detec_file)
detections.filter('time_max_offset > 787', inplace=True)
detections.filter('time_max_offset < 787.4', inplace=True)
detections.data.reset_index(drop=True,inplace=True)


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

 # Look up data files for all channels
in_file = detections.data.iloc[0]['audio_file_name'] + detections.data.iloc[0]['audio_file_extension']
audio_files = tools.find_audio_files(in_file, hydrophones_config)

print("Localization")
localizations = localization.GridSearch.run_localization(
    audio_files,
    detections,
    tdoa_grid,
    deployment_info_file,
    detection_config,
    hydrophones_config,
    localization_config,
    verbose=False,
)
 