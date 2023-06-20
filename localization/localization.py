# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:42:59 2022

@author: xavier.mouy
"""

import pandas as pd
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
import time
from ecosound.core.audiotools import upsample
import scipy.signal
from tqdm import tqdm

# from numba import njit
from numba import njit, prange
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import matplotlib.cm

# import scipy.spatial
import numpy as np
import datetime
import math

from ecosound.core.audiotools import Sound, upsample
from ecosound.core.spectrogram import Spectrogram
from ecosound.core.measurement import Measurement
from ecosound.detection.detector_builder import DetectorFactory
from ecosound.visualization.grapher_builder import GrapherFactory
import ecosound.core.tools
from ecosound.core.tools import derivative_1d, envelope, read_yaml

# from localizationlib import euclidean_dist, calc_hydrophones_distances, calc_tdoa, defineReceiverPairs, defineJacobian, predict_tdoa, linearized_inversion, solve_iterative_ML, defineCubeVolumeGrid, defineSphereVolumeGrid
import platform
import matplotlib.pyplot as plt
import tools
import mpl_toolkits.mplot3d

import dask.array as da


class LinearizedInversion:
    @staticmethod
    # @njit(parallel=False)
    def defineJacobian(R, S, V, Rpairs):
        # unknowns = ['x','y','z']    # unknowns: 3D coordinates of sound source
        N = R.shape[0] - 1  # nb of measurements (TDOAs)
        M = S.shape[1]  # number of model parameters (unknowns)
        nsources = S.shape[0]  # number of sources
        J = [None] * nsources  # initiaization
        # for each source location
        for idx in range(nsources):
            # s = S.iloc[idx]
            # j = np.full([N, M], np.nan)  # initialization of Jacobian for that source location
            j = np.zeros((N, M), dtype=np.float64)
            for i in range(N):
                p1 = Rpairs[i, 0]  # receiver #1 ID
                p2 = Rpairs[i, 1]  # receiver #2 ID
                for kk in range(M):
                    Term1 = (
                        (1 / V)
                        * 0.5
                        * (
                            (
                                ((S[idx, 0] - R[p1, 0]) ** 2)
                                + ((S[idx, 1] - R[p1, 1]) ** 2)
                                + ((S[idx, 2] - R[p1, 2]) ** 2)
                            )
                            ** (-0.5)
                        )
                        * 2
                        * (S[idx, kk] - R[p1, kk])
                    )
                    Term2 = (
                        (1 / V)
                        * 0.5
                        * (
                            (
                                ((S[idx, 0] - R[p2, 0]) ** 2)
                                + ((S[idx, 1] - R[p2, 1]) ** 2)
                                + ((S[idx, 2] - R[p2, 2]) ** 2)
                            )
                            ** (-0.5)
                        )
                        * 2
                        * (S[idx, kk] - R[p2, kk])
                    )
                    j[i][kk] = Term2 - Term1
                # for kk, unknown in enumerate(unknowns):
                #     Term1 = (1/V)*0.5*((((s.x-R.x[p1])**2)+((s.y-R.y[p1])**2)+((s.z-R.z[p1])**2))**(-0.5))*2*(s[unknown]-R[unknown][p1])
                #     Term2 = (1/V)*0.5*((((s.x-R.x[p2])**2)+((s.y-R.y[p2])**2)+((s.z-R.z[p2])**2))**(-0.5))*2*(s[unknown]-R[unknown][p2])
                #     j[i][kk] = Term2 - Term1
            J[idx] = j  # stacks jacobians for each source
        if nsources == 1:
            J = J[0]
        return J

    @staticmethod
    def getUncertainties(J, NoiseVariance):
        nsources = len(J)
        errLoc_X = [None] * nsources
        errLoc_Y = [None] * nsources
        errLoc_Z = [None] * nsources
        errLoc_RMS = [None] * nsources
        for i in range(nsources):
            Cm = NoiseVariance * np.linalg.inv(
                np.dot(np.transpose(J[i]), J[i])
            )  # covariance matrix of the model
            errLoc_X[i], errLoc_Y[i], errLoc_Z[i] = np.sqrt(
                np.diag(Cm)
            )  # uncertainty (std) along each axis
            errLoc_RMS[i] = np.sqrt(
                errLoc_X[i] ** 2 + errLoc_Y[i] ** 2 + errLoc_Z[i] ** 2
            )  # overall uncertainty (RMS)
        Uncertainty = pd.DataFrame(
            {"x": errLoc_X, "y": errLoc_Y, "z": errLoc_Z, "rms": errLoc_RMS}
        )
        return Uncertainty

    @staticmethod
    def getCost(R, S, Rpairs, V, NoiseVariance):
        # Get list of Jacobian matrice for each source
        J = LinearizedInversion.defineJacobian(R, S, V, Rpairs)
        # Calculates localization uncertainty for each source
        Uncertainties = LinearizedInversion.getUncertainties(J, NoiseVariance)
        # Get max uncertainty
        # E = max(Uncertainties.rms)
        E = np.mean(Uncertainties.rms)
        # E = np.median(Uncertainties.rms)
        return E

    @staticmethod
    def solve_iterative_ML(
        d,
        hydrophones_coords,
        hydrophone_pairs,
        m,
        V,
        damping_factor,
        verbose=True,
    ):
        # Define the Jacobian matrix: Eq. (5) in Mouy et al. (2018)
        A = LinearizedInversion.defineJacobian(
            hydrophones_coords, m, V, hydrophone_pairs
        )
        # Predicted TDOA at m (forward problem): Eq. (1) in Mouy et al. (2018)
        d0 = predict_tdoa(m, V, hydrophones_coords, hydrophone_pairs)
        # Reformulation of the problem
        delta_d = d - d0  # Delta d: measured data - predicted data
        try:
            # Resolving by creeping approach(ML inverse for delta m): Eq. (6) in Mouy et al. (2018)
            delta_m = np.dot(
                np.dot(np.linalg.inv(np.dot(A.transpose(), A)), A.transpose()),
                delta_d,
            )  # general inverse
            # New model: Eq. (7) in Mouy et al. (2018)
            # m_new = m.iloc[0].values + (damping_factor*delta_m.transpose())
            m = m + (damping_factor * delta_m.transpose())
            # m['x'] = m_new[0,0]
            # m['y'] = m_new[0,1]
            # m['z'] = m_new[0,2]
            # ## jumping approach
            # #m=inv(A'*CdInv*A)*A'*CdInv*dprime; % retrieved model using ML
            # Data misfit
            part1 = np.dot(A, delta_m) - delta_d
            data_misfit = np.dot(part1.transpose(), part1)
        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                if verbose:
                    print("Error when inverting for delta_m: Singular Matrix")
                # m['x'] = np.nan
                # m['y'] = np.nan
                # m['z'] = np.nan
                m = [[np.nan, np.nan, np.nan]]
                data_misfit = np.array([[np.nan]])
            else:
                raise err
        return np.array(m), data_misfit[0, 0]

    @staticmethod
    def calc_data_error(
        tdoa_sec, m, sound_speed_mps, hydrophones_config, hydrophone_pairs
    ):
        """Calculates tdoa measurement errors. Eq. (9) in Mouy et al. 2018"""
        hydrophones_coord = hydrophones_config[["x", "y", "z"]].to_numpy()
        tdoa_m = predict_tdoa(
            m, sound_speed_mps, hydrophones_coord, np.array(hydrophone_pairs)
        )
        Q = m.shape[0]  # nb of localizations
        M = m.shape[1]  # nb of unkmowns
        N = len(tdoa_sec)  # nb of data
        error_std = np.sqrt(
            (1 / (Q * (N - M))) * (sum((tdoa_sec - tdoa_m) ** 2))
        )
        return error_std

    @staticmethod
    def calc_loc_errors(
        tdoa_errors_std,
        m,
        sound_speed_mps,
        hydrophones_config,
        hydrophone_pairs,
    ):
        """Calculates localization errors. Eq. (8) in Mouy et al. 2018."""
        hydrophones_coord = hydrophones_config[["x", "y", "z"]].to_numpy()
        A = LinearizedInversion.defineJacobian(
            hydrophones_coord, m, sound_speed_mps, np.array(hydrophone_pairs)
        )
        Cm = (tdoa_errors_std**2) * np.linalg.inv(
            np.dot(A.transpose(), A)
        )  # Model covariance matrix for IID
        err_std = np.sqrt(np.diag(Cm))
        return pd.DataFrame(
            {
                "x_std": [err_std[0]],
                "y_std": [err_std[1]],
                "z_std": [err_std[2]],
            }
        )

    @staticmethod
    def solve(
        d,
        hydrophones_coords,
        hydrophone_pairs,
        inversion_params,
        sound_speed_mps,
        verbose=True,
        doplot=False,
    ):  # linearized_inversion(d, hydrophones_coords,hydrophone_pairs,inversion_params, sound_speed_mps, doplot=False):
        # convert parameters to numpy arrays
        damping_factor = np.array(inversion_params["damping_factor"])
        Tdelta_m = np.array(inversion_params["stop_delta_m"])
        V = np.array(sound_speed_mps)
        max_iteration = np.array(inversion_params["stop_max_iteration"])

        hydrophones_coords_np = hydrophones_coords[["x", "y", "z"]].to_numpy()
        hydrophone_pairs_np = np.array(hydrophone_pairs)

        # define starting model(s)
        start_models = [inversion_params["start_model"]]
        if inversion_params["start_model_repeats"] > 1:
            x_bounds = [
                hydrophones_coords["x"].min(),
                hydrophones_coords["x"].max(),
            ]
            y_bounds = [
                hydrophones_coords["y"].min(),
                hydrophones_coords["y"].max(),
            ]
            z_bounds = [
                hydrophones_coords["z"].min(),
                hydrophones_coords["z"].max(),
            ]
            for m_nb in range(1, inversion_params["start_model_repeats"]):
                x_tmp = np.random.uniform(low=x_bounds[0], high=x_bounds[1])
                y_tmp = np.random.uniform(low=y_bounds[0], high=y_bounds[1])
                z_tmp = np.random.uniform(low=z_bounds[0], high=z_bounds[1])
                start_models.append([x_tmp, y_tmp, z_tmp])
        m_stack = []
        iterations_logs_stack = []
        data_misfit_stack = []
        for start_model in start_models:
            # current starting model
            # m = pd.DataFrame({'x': [start_model[0]], 'y': [start_model[1]], 'z': [start_model[2]]})
            m = np.array([[start_model[0], start_model[1], start_model[2]]])
            # Keeps track of values for each iteration
            iterations_logs = pd.DataFrame(
                {
                    "x": [start_model[0]],
                    "y": [start_model[1]],
                    "z": [start_model[2]],
                    "norm": np.nan,
                    "data_misfit": np.nan,
                }
            )
            # Start iterations
            stop = False
            idx = 0
            while stop == False:
                idx = idx + 1
                # print(idx)
                # Linear inversion
                m_it, data_misfit_it = LinearizedInversion.solve_iterative_ML(
                    d,
                    hydrophones_coords_np,
                    hydrophone_pairs_np,
                    m,
                    V,
                    damping_factor,
                    verbose=verbose,
                )
                # Save model and data misfit for each iteration
                # iterations_logs = iterations_logs.append({
                #     'x': m_it['x'].values[0],
                #     'y': m_it['y'].values[0],
                #     'z': m_it['z'].values[0],
                #     #'norm': np.sqrt(np.square(m_it).sum(axis=1)).values[0],
                #     'norm': np.linalg.norm(m_it.iloc[0].to_numpy() - iterations_logs.iloc[-1][['x','y','z']].to_numpy(),np.inf),
                #     'data_misfit': data_misfit_it,
                #     }, ignore_index=True)

                # iterations_logs = pd.concat(
                #     [iterations_logs,
                #      pd.DataFrame(data = {'x': [m_it['x'].values[0]],
                #                           'y': [m_it['y'].values[0]],
                #                           'z': [m_it['z'].values[0]],
                #                           #'norm': np.sqrt(np.square(m_it).sum(axis=1)).values[0],
                #                           'norm': [np.linalg.norm(m_it.iloc[0].to_numpy() - iterations_logs.iloc[-1][['x','y','z']].to_numpy(),np.inf)],
                #                           'data_misfit': [data_misfit_it],
                #                           }
                #                   )
                #      ],
                #     ignore_index=True)

                iterations_logs = pd.concat(
                    [
                        iterations_logs,
                        pd.DataFrame(
                            data={
                                "x": [m_it[0, 0]],
                                "y": [m_it[0, 1]],
                                "z": [m_it[0, 2]],
                                #'norm': np.sqrt(np.square(m_it).sum(axis=1)).values[0],
                                "norm": [
                                    np.linalg.norm(
                                        m_it
                                        - iterations_logs.iloc[-1][
                                            ["x", "y", "z"]
                                        ].to_numpy(),
                                        np.inf,
                                    )
                                ],
                                "data_misfit": [data_misfit_it],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

                # Update m
                m = m_it
                # stopping criteria
                if idx > 1:
                    # norm of model diference
                    # if (iterations_logs['norm'][idx] - iterations_logs['norm'][idx-1] <= Tdelta_m):
                    if iterations_logs["norm"][idx] <= Tdelta_m:
                        stop = True
                        # if np.isnan(m['x'].values[0]):
                        if np.isnan(m[0, 0]):
                            converged = False  # Due to singular matrix
                            if verbose:
                                print(
                                    "Singular matrix - inversion hasn"
                                    "t converged."
                                )
                        else:
                            converged = True
                    elif idx > max_iteration:  # max iterations exceeded
                        stop = True
                        converged = False
                        if verbose:
                            print(
                                "Max iterations reached - inversion hasn"
                                "t converged."
                            )
                    # elif iterations_logs['norm'][idx] > iterations_logs['norm'][idx-1]: # if norm starts increasing -> then stop there
                    #     stop = True
                    #     converged=False
                    #     print('Norm increasing - inversion hasn''t converged.')
                    # elif np.isnan(m['x'].values[0]):
                    elif np.isnan(m[0, 0]):
                        stop = True
                        converged = False
                        if verbose:
                            print(
                                "Singular matrix - inversion hasn"
                                "t converged."
                            )

            if not converged:
                # m['x']=np.nan
                # m['y']=np.nan
                # m['z']=np.nan
                m[0, 0] = np.nan
                m[0, 1] = np.nan
                m[0, 2] = np.nan
            # Save results for each starting model
            m_stack.append(m[0, :])
            iterations_logs_stack.append(iterations_logs)
            data_misfit_stack.append(iterations_logs["data_misfit"].iloc[-1])
        # Select results for starting model with best ending data misfit
        min_data_misfit = np.nanmin(data_misfit_stack)
        if np.isnan(min_data_misfit):
            min_idx = 0
        else:
            min_idx = data_misfit_stack.index(min_data_misfit)
        return m_stack[min_idx], iterations_logs_stack[min_idx]

    @staticmethod
    def run_localization(
        audio_files,
        detections,
        deployment_info,
        detection_config,
        hydrophones_config,
        localization_config,
        verbose=True,
    ):

        # localization
        sound_speed_mps = localization_config["ENVIRONMENT"]["sound_speed_mps"]
        ref_channel = localization_config["TDOA"]["ref_channel"]

        # define search window based on hydrophone separation and sound speed
        hydrophones_dist_matrix = tools.calc_hydrophones_distances(
            hydrophones_config
        )
        TDOA_max_sec = np.max(hydrophones_dist_matrix) / sound_speed_mps

        # define hydrophone pairs
        hydrophone_pairs = defineReceiverPairs(
            len(hydrophones_config), ref_receiver=ref_channel
        )

        # Prepare localization object
        localizations = Measurement()
        localizations.metadata["measurer_name"] = "Linearized inversion"
        localizations.metadata["measurer_version"] = "0.1"
        localizations.metadata["measurements_name"] = [
            [
                "x_m",
                "y_m",
                "z_m",
                "x_err_low_m",
                "x_err_high_m",
                "x_err_span_m",
                "y_err_low_m",
                "y_err_high_m",
                "y_err_span_m",
                "z_err_low_m",
                "z_err_high_m",
                "z_err_span_m",
                "tdoa_errors_std_sec",
            ]
        ]

        # Go through all detections
        # print('LOCALIZATION')
        for detec_idx, detec in detections.data.iterrows():

            # print( str(detec_idx+1) + '/' + str(len(detections)))

            # load data from all channels for that detection
            waveform_stack = tools.stack_waveforms(
                audio_files, detec, TDOA_max_sec
            )

            # readjust signal boundaries to only focus on section with most energy
            percentage_max_energy = 90
            chunk = ecosound.core.tools.tighten_signal_limits_peak(
                waveform_stack[detection_config["AUDIO"]["channel"]],
                percentage_max_energy,
            )
            waveform_stack = [x[chunk[0] : chunk[1]] for x in waveform_stack]

            # calculate TDOAs
            tdoa_sec, corr_val = calc_tdoa(
                waveform_stack,
                hydrophone_pairs,
                detec["audio_sampling_frequency"],
                TDOA_max_sec=TDOA_max_sec,
                upsample_res_sec=localization_config["TDOA"][
                    "upsample_res_sec"
                ],
                normalize=localization_config["TDOA"]["normalize"],
                doplot=False,
            )

            valid_tdoas = (
                np.min(corr_val) > localization_config["TDOA"]["min_corr_val"]
            )

            if valid_tdoas:
                # Lineralized inversion
                [m, iterations_logs] = LinearizedInversion.solve(
                    tdoa_sec,
                    hydrophones_config,
                    hydrophone_pairs,
                    localization_config["INVERSION"],
                    sound_speed_mps,
                    verbose=verbose,
                    doplot=False,
                )
                m = np.array([m])

                # Estimate uncertainty
                tdoa_errors_std = LinearizedInversion.calc_data_error(
                    tdoa_sec,
                    m,
                    sound_speed_mps,
                    hydrophones_config,
                    hydrophone_pairs,
                )
                loc_errors_std = LinearizedInversion.calc_loc_errors(
                    tdoa_errors_std,
                    m,
                    sound_speed_mps,
                    hydrophones_config,
                    hydrophone_pairs,
                )

                # Bring all detection and localization informations together
                detec.loc["x_m"] = m[0, 0]
                detec.loc["y_m"] = m[0, 1]
                detec.loc["z_m"] = m[0, 2]
                detec.loc["x_err_low_m"] = (
                    detec.loc["x_m"] - loc_errors_std["x_std"].values[0]
                )
                detec.loc["x_err_high_m"] = (
                    detec.loc["x_m"] + loc_errors_std["x_std"].values[0]
                )
                detec.loc["x_err_span_m"] = (
                    detec.loc["x_err_high_m"] - detec.loc["x_err_low_m"]
                )
                detec.loc["y_err_low_m"] = (
                    detec.loc["y_m"] - loc_errors_std["y_std"].values[0]
                )
                detec.loc["y_err_high_m"] = (
                    detec.loc["y_m"] + loc_errors_std["y_std"].values[0]
                )
                detec.loc["y_err_span_m"] = (
                    detec.loc["y_err_high_m"] - detec.loc["y_err_low_m"]
                )
                detec.loc["z_err_low_m"] = (
                    detec.loc["z_m"] - loc_errors_std["z_std"].values[0]
                )
                detec.loc["z_err_high_m"] = (
                    detec.loc["z_m"] + loc_errors_std["z_std"].values[0]
                )
                detec.loc["z_err_span_m"] = (
                    detec.loc["z_err_high_m"] - detec.loc["z_err_low_m"]
                )
                detec.loc["tdoa_errors_std_sec"] = tdoa_errors_std[0]
                detec.loc["iterations_logs"] = iterations_logs
                detec = detec.to_frame().T
            else:  # detections that couln'd be localized (low xcorr value for tdoa measurements)
                detec.loc["x_m"] = np.nan
                detec.loc["y_m"] = np.nan
                detec.loc["z_m"] = np.nan
                detec.loc["x_err_low_m"] = np.nan
                detec.loc["x_err_high_m"] = np.nan
                detec.loc["x_err_span_m"] = np.nan
                detec.loc["y_err_low_m"] = np.nan
                detec.loc["y_err_high_m"] = np.nan
                detec.loc["y_err_span_m"] = np.nan
                detec.loc["z_err_low_m"] = np.nan
                detec.loc["z_err_high_m"] = np.nan
                detec.loc["z_err_span_m"] = np.nan
                detec.loc["tdoa_errors_std_sec"] = np.nan
                detec.loc["iterations_logs"] = np.nan
                detec = detec.to_frame().T

            # stack to results into localization object
            if len(localizations) == 0:
                localizations.data = detec
            else:
                localizations.data = pd.concat(
                    [localizations.data, detec], axis=0
                )
        print("...done")
        return localizations


class GridSearch:
    @staticmethod
    def calc_data_error(
        tdoa_sec, m, sound_speed_mps, hydrophones_config, hydrophone_pairs
    ):
        """Calculates tdoa measurement errors. Eq. (9) in Mouy et al. 2018"""
        tdoa_m = predict_tdoa(
            m, sound_speed_mps, hydrophones_config, hydrophone_pairs
        )
        Q = len(m)
        M = m.size
        N = len(tdoa_sec)
        error_std = np.sqrt(
            (1 / (Q * (N - M))) * (sum((tdoa_sec - tdoa_m) ** 2))
        )
        return error_std

    @staticmethod
    def create_tdoa_grid(
        x_limits,
        y_limits,
        z_limits,
        spacing,
        hydrophones_config,
        ref_channel,
        sound_speed_mps,
        outfile,
    ):
        hydrophone_pairs = defineReceiverPairs(
            len(hydrophones_config), ref_receiver=ref_channel
        )
        sources = tools.defineCubeVolumeGrid(
            x_limits, y_limits, z_limits, spacing
        )
        sources_tdoa = np.zeros(shape=(len(hydrophone_pairs), len(sources)))
        points_coord = sources.to_numpy()
        hydrophones_coord = hydrophones_config[["x", "y", "z"]].to_numpy()
        tdoas = predict_tdoa(
            points_coord,
            sound_speed_mps,
            hydrophones_coord,
            np.array(hydrophone_pairs),
        )
        np.savez(
            outfile,
            tdoa=tdoas,
            grid_coord=sources,
            x_limits=x_limits,
            y_limits=y_limits,
            z_limits=z_limits,
            spacing=spacing,
            ref_channel=ref_channel,
            sound_speed_mps=sound_speed_mps,
            hydrophones_coord=hydrophones_coord,
        )
        # print("...done")
        # return sources, sources_tdoa

    @staticmethod
    def load_tdoa_grid(grid_file):
        npzfile = np.load(grid_file)
        tdoa_grid = dict()
        tdoa_grid["tdoa"] = npzfile["tdoa"]
        tdoa_grid["grid_coord"] = npzfile["grid_coord"]
        tdoa_grid["x_limits"] = npzfile["x_limits"]
        tdoa_grid["y_limits"] = npzfile["y_limits"]
        tdoa_grid["z_limits"] = npzfile["z_limits"]
        tdoa_grid["spacing"] = npzfile["spacing"]
        tdoa_grid["ref_channel"] = npzfile["ref_channel"]
        tdoa_grid["sound_speed_mps"] = npzfile["sound_speed_mps"]
        tdoa_grid["hydrophones_coord"] = npzfile["hydrophones_coord"]
        return tdoa_grid

    @staticmethod
    def calc_credibility_interval(axis, values, start_value, percentage):
        values = values / sum(values)
        start_index = int(np.where(axis == start_value)[0])
        IC_low_limit_idx, remainder = find_half_CI_value(
            values, start_index, percentage / 2, -1
        )
        IC_high_limit_idx, remainder = find_half_CI_value(
            values, start_index + 1, percentage / 2 + remainder, 1
        )
        if remainder > 0:
            IC_low_limit_idx, remainder = find_half_CI_value(
                values, IC_low_limit_idx, remainder, -1
            )
        if sum(values[IC_low_limit_idx : IC_high_limit_idx + 1]) < percentage:
            print("Issue while calculating the CI")
        return [axis[IC_low_limit_idx], axis[IC_high_limit_idx]]

    @staticmethod
    def run_localization(
        audio_files,
        detections,
        tdoa_grid,
        deployment_info,
        detection_config,
        hydrophones_config,
        localization_config,
        verbose=True,
    ):

        # localization
        sound_speed_mps = localization_config["ENVIRONMENT"]["sound_speed_mps"]
        ref_channel = localization_config["TDOA"]["ref_channel"]

        # define search window based on hydrophone separation and sound speed
        hydrophones_dist_matrix = tools.calc_hydrophones_distances(
            hydrophones_config
        )
        TDOA_max_sec = np.max(hydrophones_dist_matrix) / sound_speed_mps

        # define hydrophone pairs
        hydrophone_pairs = defineReceiverPairs(
            len(hydrophones_config), ref_receiver=ref_channel
        )

        # Prepare localization object
        localizations = Measurement()
        localizations.metadata["measurer_name"] = "Grid Search"
        localizations.metadata["measurer_version"] = "0.1"
        localizations.metadata["measurements_name"] = [
            [
                "x_m",
                "y_m",
                "z_m",
                "x_err_low_m",
                "x_err_high_m",
                "x_err_span_m",
                "y_err_low_m",
                "y_err_high_m",
                "y_err_span_m",
                "z_err_low_m",
                "z_err_high_m",
                "z_err_span_m",
                "tdoa_sec",
                "tdoa_errors_std_sec",
                "best_freq_band_hz",
                "corr_val",
            ]
        ]

        # print('LOCALIZATION')
        # Go through all detections and resolve localization
        m = np.zeros([3, len(detections)])
        tdoa_errors = np.zeros([len(hydrophone_pairs), len(detections)])
        grid_data_misfit = [[]] * len(detections)
        tdoa_xcorr_vals = np.zeros([len(hydrophone_pairs), len(detections)])
        PPDs = [[]] * len(detections)
        
        modelled_tdoa_grid = da.from_array(tdoa_grid["tdoa"],chunks=100000)
        
        for detec_idx, detec in tqdm(
            detections.data.iterrows(),
            total=len(detections),
            desc=" localizing source",
            leave=True,
            miniters=1,
            colour="green",
        ):

            # define the different search frequency bands
            number_freq_bands = localization_config["TDOA"]["number_freq_bands"]#4
            min_bandwidth_hz = localization_config["TDOA"]["min_bandwidth_hz"]#4#200
            freq_bands = split_freq_bands(detec["frequency_min"], detec["frequency_max"], number_freq_bands, min_bandwidth_hz)
            
            tdoa_sec_stack = []
            corr_val_stack = []
            min_delta_tdoa_norm_stack = []
            min_idx_stack = []
            delta_tdoa_stack = [] 
            delta_tdoa_norm_stack = []
            for freq_band in freq_bands: # loop through different freq bands
                
                detec["frequency_min"] = freq_band[0]
                detec["frequency_max"] = freq_band[1]
                
                # load data from all channels for that detection
                waveform_stack = tools.stack_waveforms(
                    audio_files, detec, TDOA_max_sec
                )
    
                # readjust signal boundaries to only focus on section with most energy
                #percentage_max_energy = 80
                percentage_max_energy = localization_config["TDOA"]["energy_window_perc"]
                # chunk = ecosound.core.tools.tighten_signal_limits_peak(
                #     waveform_stack[detection_config["AUDIO"]["channel"]],
                #     percentage_max_energy,
                # )
                
                chunk = ecosound.core.tools.tighten_signal_limits(
                    waveform_stack[detection_config["AUDIO"]["channel"]],
                    percentage_max_energy,
                )                
                           
                
                # fig, ax = plt.subplots(nrows=6,ncols=1)
                # ax[0].plot(waveform_stack[0])
                # ax[1].plot(waveform_stack[1])
                # ax[2].plot(waveform_stack[2])
                # ax[3].plot(waveform_stack[3])
                # ax[4].plot(waveform_stack[4])
                # ax[5].plot(waveform_stack[5])
                # #ax.set_title('Simple plot')
                # plt.show()
                
                waveform_stack = [x[chunk[0] : chunk[1]] for x in waveform_stack] 
                
                # fig, ax = plt.subplots(nrows=6,ncols=1)
                # ax[0].plot(waveform_stack[0])
                # ax[1].plot(waveform_stack[1])
                # ax[2].plot(waveform_stack[2])
                # ax[3].plot(waveform_stack[3])
                # ax[4].plot(waveform_stack[4])
                # ax[5].plot(waveform_stack[5])
                # #ax.set_title('Simple plot')
                # plt.show()
                

                # calculate TDOAs
                tdoa_sec, corr_val = calc_tdoa(
                    waveform_stack,
                    hydrophone_pairs,
                    detec["audio_sampling_frequency"],
                    TDOA_max_sec=TDOA_max_sec,
                    upsample_res_sec=localization_config["TDOA"][
                        "upsample_res_sec"
                    ],
                    normalize=localization_config["TDOA"]["normalize"],
                    doplot=False,
                )
                
                # Find grid location minimizing the data misfit
                delta_tdoa = modelled_tdoa_grid - tdoa_sec
                delta_tdoa_norm = da.linalg.norm(delta_tdoa, axis=0)
                min_idx = da.argmin(delta_tdoa_norm)
                min_delta_tdoa_norm = delta_tdoa_norm[min_idx]
                
                # stack values for each frequency band tested
                tdoa_sec_stack.append(tdoa_sec)
                corr_val_stack.append(corr_val)
                delta_tdoa_stack.append(delta_tdoa)
                delta_tdoa_norm_stack.append(delta_tdoa_norm)
                min_idx_stack.append(min_idx.compute())
                min_delta_tdoa_norm_stack.append(min_delta_tdoa_norm.compute())

            
            # decide best frequency band (the one minimizing the data misfit)
            best_freqband_idx = np.argmin(min_delta_tdoa_norm_stack)
            best_freqband = freq_bands[best_freqband_idx]
            tdoa_sec = tdoa_sec_stack[best_freqband_idx]
            corr_val = corr_val_stack[best_freqband_idx]
            delta_tdoa = delta_tdoa_stack[best_freqband_idx]
            delta_tdoa_norm = delta_tdoa_norm_stack[best_freqband_idx]
            min_idx = min_idx_stack[best_freqband_idx]
            min_delta_tdoa_norm = min_delta_tdoa_norm_stack [best_freqband_idx]
            
            # clear intermediate _stack variable to save memory
            tdoa_sec_stack = []
            corr_val_stack = []
            min_delta_tdoa_norm_stack = []
            min_idx_stack = []
            delta_tdoa_stack = [] 
            delta_tdoa_norm_stack = []
            
            
            if np.min(corr_val) > localization_config["TDOA"]["min_corr_val"]:
            
                # Solution
                m = tdoa_grid["grid_coord"][min_idx].T
                
                # Estimate data errors            
                N=1
                tdoa_errors_std = np.sqrt(sum(delta_tdoa[:, min_idx]**2) / (np.prod(delta_tdoa[:, min_idx].shape)-N))
                
                # Unnormalized likelihood
                L = np.exp(
                    -delta_tdoa_norm ** 2
                    / (2 * (tdoa_errors_std**2))
                )
                # Normalized PPD
                PPD = L / L.sum()            
                
                # Convert linear array into dataframe then 3D numpy array
                PPD_df = pd.DataFrame({"x": [], "y": [], "z": [], "PPD": []})
                PPD_df[["x", "y", "z"]] = tdoa_grid["grid_coord"]
                PPD_df["PPD"] = PPD
                PPD_df = PPD_df.set_index(["x", "y", "z"])
                PPD_xr = PPD_df.to_xarray()
    
                # calculate 2D marginals
                # Pxy = PPD_xr.PPD.sum("z")
                # Pxz = PPD_xr.PPD.sum("y")
                # Pyz = PPD_xr.PPD.sum("x")
    
                # calculate 1D marginals
                Px = PPD_xr.PPD.sum("z").sum("y")
                Py = PPD_xr.PPD.sum("x").sum("z")
                Pz = PPD_xr.PPD.sum("x").sum("y")
    
                # calculate credibility intervals from 1D marginals
                percentage = 0.68  # % for CI
                Px_CI = GridSearch.calc_credibility_interval(
                    Px["x"].values, Px.to_numpy(), m[0], percentage
                )
                Py_CI = GridSearch.calc_credibility_interval(
                    Py["y"].values, Py.to_numpy(), m[1], percentage
                )
                Pz_CI = GridSearch.calc_credibility_interval(
                    Pz["z"].values, Pz.to_numpy(), m[2], percentage
                )
    
                # Bring all detection and localization informations together
                detec.loc["x_m"] = m[0]
                detec.loc["y_m"] = m[1]
                detec.loc["z_m"] = m[2]
                detec.loc["x_err_low_m"] = Px_CI[0]
                detec.loc["x_err_high_m"] = Px_CI[1]
                detec.loc["x_err_span_m"] = (
                    detec.loc["x_err_high_m"] - detec.loc["x_err_low_m"]
                )
                detec.loc["y_err_low_m"] = Py_CI[0]
                detec.loc["y_err_high_m"] = Py_CI[1]
                detec.loc["y_err_span_m"] = (
                    detec.loc["y_err_high_m"] - detec.loc["y_err_low_m"]
                )
                detec.loc["z_err_low_m"] = Pz_CI[0]
                detec.loc["z_err_high_m"] = Pz_CI[1]
                detec.loc["z_err_span_m"] = (
                    detec.loc["z_err_high_m"] - detec.loc["z_err_low_m"]
                )
                detec.loc["tdoa_errors_std_sec"] = tdoa_errors_std.compute()
                detec.loc["tdoa_sec"] = str(tdoa_sec.T[0])
                detec.loc["best_freq_band_hz"] = str(best_freqband)
                detec.loc["corr_val"] = str(corr_val.T[0])
                detec.loc["frequency_min"] = detections.data["frequency_min"].iloc[detec_idx]
                detec.loc["frequency_max"] = detections.data["frequency_max"].iloc[detec_idx]
                
                # detec.loc['PPD'] = PPD_xr
                detec = detec.to_frame().T
                # save PPDs
                #PPDs[detec_idx] = PPD_xr
            else:
                
                detec.loc["x_m"] = np.nan
                detec.loc["y_m"] = np.nan
                detec.loc["z_m"] = np.nan
                detec.loc["x_err_low_m"] = np.nan
                detec.loc["x_err_high_m"] = np.nan
                detec.loc["x_err_span_m"] = np.nan
                detec.loc["y_err_low_m"] = np.nan
                detec.loc["y_err_high_m"] = np.nan
                detec.loc["y_err_span_m"] = np.nan
                detec.loc["z_err_low_m"] = np.nan
                detec.loc["z_err_high_m"] = np.nan
                detec.loc["z_err_span_m"] = np.nan
                detec.loc["tdoa_errors_std_sec"] = np.nan
                detec.loc["tdoa_sec"] = str(tdoa_sec.T[0])
                detec.loc["best_freq_band_hz"] = ''
                detec.loc["corr_val"] = str(corr_val.T[0])
                detec.loc["frequency_min"] = detections.data["frequency_min"].iloc[detec_idx]
                detec.loc["frequency_max"] = detections.data["frequency_max"].iloc[detec_idx]
                # detec.loc['PPD'] = np.nan
                detec = detec.to_frame().T

            # stack to results into localization object
            if len(localizations) == 0:
                localizations.data = detec
            else:
                localizations.data = pd.concat(
                    [localizations.data, detec], axis=0
                )
        # print("...done")

        return localizations


def defineReceiverPairs(n_receivers, ref_receiver=0):
    Rpairs = []
    for i in range(n_receivers):
        if i != ref_receiver:
            pair = [ref_receiver, i]
            Rpairs.append(pair)
    return Rpairs


def predict_tdoa_old(m, V, hydrophones_coords, hydrophone_pairs):
    """Create data from the forward problem.
    Generate TDOAs based on location of hydrophones and source.
    """
    N = len(hydrophone_pairs)
    dt = np.full([N, 1], np.nan)
    for idx, hydrophone_pair in enumerate(hydrophone_pairs):
        p1 = hydrophone_pair[0]
        p2 = hydrophone_pair[1]
        t1 = (1 / V) * np.sqrt(
            ((m.x - hydrophones_coords.x[p1]) ** 2)
            + ((m.y - hydrophones_coords.y[p1]) ** 2)
            + ((m.z - hydrophones_coords.z[p1]) ** 2)
        )
        t2 = (1 / V) * np.sqrt(
            ((m.x - hydrophones_coords.x[p2]) ** 2)
            + ((m.y - hydrophones_coords.y[p2]) ** 2)
            + ((m.z - hydrophones_coords.z[p2]) ** 2)
        )
        dt[idx] = np.array(t2 - t1)  # noiseless data
    return dt


@njit(parallel=False)
def predict_tdoa(m, V, hydrophones_coords, hydrophones_pairs):
    """Create data from the forward problem.
    Generate TDOAs based on location of hydrophones and source.
    """
    N = hydrophones_pairs.shape[0]
    n_sources = m.shape[0]
    dt = np.zeros((N, n_sources), dtype=np.float64)
    for m_idx in prange(n_sources):
        for pair_idx in prange(N):
            p1 = hydrophones_pairs[pair_idx, 0]
            p2 = hydrophones_pairs[pair_idx, 1]
            t1 = (1 / V) * np.sqrt(
                ((m[m_idx, 0] - hydrophones_coords[p1, 0]) ** 2)
                + ((m[m_idx, 1] - hydrophones_coords[p1, 1]) ** 2)
                + ((m[m_idx, 2] - hydrophones_coords[p1, 2]) ** 2)
            )
            t2 = (1 / V) * np.sqrt(
                ((m[m_idx, 0] - hydrophones_coords[p2, 0]) ** 2)
                + ((m[m_idx, 1] - hydrophones_coords[p2, 1]) ** 2)
                + ((m[m_idx, 2] - hydrophones_coords[p2, 2]) ** 2)
            )
            dt[pair_idx, m_idx] = t2 - t1  # noiseless data
    return dt


def calc_tdoa(
    waveform_stack,
    hydrophone_pairs,
    sampling_frequency,
    TDOA_max_sec=None,
    upsample_res_sec=None,
    normalize=False,
    hydrophones_name=None,
    doplot=False,
):
    """
    TDOA measurements

    Calculates the time-difference of orrival (TDOA) between signals from
    different hydrophones by cross-correlation.

    Parameters
    ----------
    waveform_stack : list of numpy arrays
        Wavforms with amplitude values of the signal for each hydrophone.
        Each wavform is a numpy array which are stored in a list
        e.g. waveform_stack[0] contains a numpy array with the wavform from
         the first hydrophone.
    hydrophone_pairs : list
        Defines the pair of hydrophones for the TDOA measurements. Each element
        of hydrophones_pairs is a list with index values of the hydrophone in
        waveform_stack.
        e.g. hydrophones_pairs = [[3, 0], [3, 1], [3, 2], [3, 4], [3, 5]].
    sampling_frequency : float
        Sampling frequency of the waveform signals in  waveform_stack in Hz.
    TDOA_max_sec : float, optional
        Restricts the TDOA search to TDOA_max_sec seconds. The default is None.
    upsample_res_sec : float, optional
        If set, upsamples the wavforms in waveform_stack before the cross-
        correlation to have a time resolution of upsample_res_sec seconds.
        The default is None.
    normalize : bool, optional
        If set to True, normalizes the wavforms in waveform_stack to have a
        maximum amplitude of 1. The default is False.
    hydrophones_name : list, optional
        list of string with the name of each hydrophone. Only used for plots.
        The default is None.
    doplot : bool, optional
        If set to True, displays cross correlation plots for each hydrophone
        pair. The default is False.

    Returns
    -------
    tdoa_sec : 1-D numpy array
        Time-difference of arrival in seconds, for each hydrophone pair.
    tdoa_corr : 1-D numpy array
        Maximum cross-correlation value for each hydrophone pair (between 0
        and 1).

    """
    tdoa_sec = []
    tdoa_corr = []
    # Upsampling
    if upsample_res_sec:
        if upsample_res_sec < (1 / sampling_frequency):
            for chan_id, waveform in enumerate(waveform_stack):
                waveform_stack[chan_id], new_sampling_frequency = upsample(
                    waveform, 1 / sampling_frequency, upsample_res_sec
                )
            sampling_frequency = new_sampling_frequency
        else:
            print(
                "Warning: upsampling not applied because the requested time"
                " resolution (upsample_res_sec) is larger than the current"
                " time resolution of the signal."
            )
    # Normalize max amplitude to 1
    if normalize:
        for chan_id, waveform in enumerate(waveform_stack):
            waveform_stack[chan_id] = waveform / np.max(waveform)
    # Constrains to a max TDOA (based on array geometry)
    if TDOA_max_sec:
        TDOA_max_samp = int(np.round(TDOA_max_sec * sampling_frequency))

    # cross correlation
    for hydrophone_pair in hydrophone_pairs:
        # signal from each hydrophone
        s1 = waveform_stack[hydrophone_pair[0]]
        s2 = waveform_stack[hydrophone_pair[1]]
        # cross correlation
        corr = scipy.signal.correlate(s2, s1, mode="full", method="auto")
        corr = corr / (np.linalg.norm(s1) * np.linalg.norm(s2))
        lag_array = scipy.signal.correlation_lags(
            s2.size, s1.size, mode="full"
        )
        # Identify correlation peak within the TDOA search window (SW)
        if TDOA_max_sec:
            tmp1 = np.where(lag_array == -TDOA_max_samp)
            if len(tmp1[0])>0:
                SW_start_idx =tmp1[0][0]  # search window start idx
            else:
                SW_start_idx = 0
            tmp2 = np.where(lag_array == TDOA_max_samp)
            if len(tmp2[0])>0:
                SW_stop_idx = tmp2[0][0]  # search window stop idx
            else:
                SW_stop_idx = len(corr) - 1 # search window stop idx
        else:
            SW_start_idx = 0
            SW_stop_idx = len(corr) - 1
        corr_max_idx = (
            np.argmax(corr[SW_start_idx:SW_stop_idx]) + SW_start_idx
        )  # idx of max corr value
        delay = lag_array[corr_max_idx]
        corr_value = corr[corr_max_idx]
        tdoa_sec.append(delay / sampling_frequency)
        tdoa_corr.append(corr_value)

        if doplot:
            fig, ax = plt.subplots(nrows=2, sharex=False)
            if hydrophones_name:
                label1 = hydrophones_name[hydrophone_pair[0]] + " (ref)"
                label2 = hydrophones_name[hydrophone_pair[1]]
            else:
                label1 = "Hydrophone " + str(hydrophone_pair[0]) + " (ref)"
                label2 = "Hydrophone " + str(hydrophone_pair[1])
            ax[0].plot(s1, color="red", label=label1)
            ax[0].plot(s2, color="black", label=label2)
            ax[0].set_xlabel("Time (sample)")
            ax[0].set_ylabel("Amplitude")
            ax[0].legend()
            ax[0].grid()
            ax[0].set_title("TDOA: " + str(delay) + " samples")
            ax[1].plot(lag_array, corr)
            ax[1].plot(delay, corr_value, marker=".", color="r", label="TDOA")
            if TDOA_max_sec:
                width = 2 * TDOA_max_samp
                height = 2
                rect = plt.Rectangle(
                    (-TDOA_max_samp, -1),
                    width,
                    height,
                    linewidth=1,
                    edgecolor="green",
                    facecolor="green",
                    alpha=0.3,
                    label="Search window",
                )
                ax[1].add_patch(rect)
            ax[1].set_xlabel("Lag (sample)")
            ax[1].set_ylabel("Correlation")
            ax[1].set_title("Correlation: " + str(corr_value))
            ax[1].set_ylim(-1, 1)
            ax[1].grid()
            ax[1].legend()
            plt.tight_layout()
    return np.array([tdoa_sec]).transpose(), np.array([tdoa_corr]).transpose()


def plot_localizations3D(localizations=None, hydrophones=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if hydrophones is not None:
        ax.scatter3D(
            hydrophones["x"],
            hydrophones["y"],
            hydrophones["z"],
            s=20,
            color="gainsboro",
            edgecolors="dimgray",
            label=hydrophones["name"],
            alpha=1,
            zorder=3,
        )

    if localizations is not None:
        loc_data = localizations.data
        ax.scatter3D(
            loc_data["x"],
            loc_data["y"],
            loc_data["z"],
            c="blue",
            # marker=params['loc_marker'].values[0],
            alpha=1,
            # s=2
        )
        # plot uncertainties
        for idx, loc_point in loc_data.iterrows():
            lw = 1
            c = "red"
            alph = 1
            ax.plot3D(
                [loc_point["x_err_low_m"], loc_point["x_err_high_m"]],
                [loc_point["y"], loc_point["y"]],
                [loc_point["z"], loc_point["z"]],
                linewidth=lw,
                # linestyle='-',
                color=c,
                alpha=alph,
            )
            ax.plot3D(
                [loc_point["x"], loc_point["x"]],
                [loc_point["y_err_low_m"], loc_point["y_err_high_m"]],
                [loc_point["z"], loc_point["z"]],
                linewidth=lw,
                # linestyle='-',
                color=c,
                alpha=alph,
            )
            ax.plot3D(
                [loc_point["x"], loc_point["x"]],
                [loc_point["y"], loc_point["y"]],
                [loc_point["z_err_low_m"], loc_point["z_err_high_m"]],
                linewidth=lw,
                # linestyle='-',
                color=c,
                alpha=alph,
            )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()
    return fig, ax


def find_half_CI_value(values, start_index, stop_val, step):

    # Regarding the credibility intervals, don't change them for your thesis,
    # but another way to do them (probably better than what I suggested
    # yesterday) is "centred" CIs which are centred in each dimension on the MAP
    # estimate. For asymmetric PPDs these aren't necessarily truly centred but
    # can be cut off in one direction if it runs out of probability (and extend
    # farther in other direction). I've attached a matlab code (update of earlier
    # code) that has an option of computing these, which you can look at when you
    # have time (not now).

    min_current_value = 0.001 * max(values)
    current_idx = start_index
    current_sum = 0
    max_idx = len(values) - 1
    while (
        (current_sum < stop_val)
        & (current_idx >= 0)
        & (current_idx <= max_idx)
    ):
        current_sum += values[current_idx]
        if values[current_idx] < min_current_value:
            break
        current_idx += step
    remainder = stop_val - current_sum
    return current_idx - step, remainder

def split_freq_bands(freq_min, freq_max, number_freq_bands, min_bandwidth_hz):

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

    return freq_bands