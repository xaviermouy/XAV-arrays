import sys
sys.path.append(r"C:\Users\xavier.mouy\Documents\GitHub\ecosound") # Adds higher directory to python modules path.

import numpy as np
import pandas as pd
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
import time
import os
from ecosound.core.audiotools import upsample
import scipy.signal
from numba import njit

def getReceiverBoundsWidth(ReceiverBounds):
    ReceiverBoundsWidth = ReceiverBounds.applymap(lambda x: max(x)-min(x))
    return ReceiverBoundsWidth


def initializeReceivers(nReceivers, ReceiverBounds):
    ReceiverBoundsWidth = getReceiverBoundsWidth(ReceiverBounds)
    R1 = [None] * ReceiverBounds.shape[1]
    R1[0] = [None] * nReceivers  # x
    R1[1] = [None] * nReceivers  # y
    R1[2] = [None] * nReceivers  # z
    for r in range(nReceivers):  # random location for each receiver and axis
        for dim in range(ReceiverBounds.shape[1]):
            R1[dim][r] = np.random.rand(1)[0]*ReceiverBoundsWidth.iloc[r][dim] + min(ReceiverBounds.iloc[r][dim])
    Receivers = pd.DataFrame({'x': R1[0], 'y': R1[1], 'z': R1[2]})
    return Receivers


def getParamsLinearMapping(R):
    Rindices = [None] * R.shape[0] * R.shape[1]
    midx = 0
    for ridx in range(R.shape[0]):
        for dimidx in range(R.shape[1]):
            Rindices[midx] = [ridx, dimidx]
            midx += 1
    return Rindices


def perturbReceivers(R, PerturbParamIdx, MappedParamsIdx, ReceiverBounds, ReceiverBoundsWidth, PerturbSTD, T0, T):
    # goes back to first parameters if reached the end of the list of parameters
    if PerturbParamIdx > len(MappedParamsIdx)-1:
        PerturbParamIdx = 0
    # Identifies from MappedParamsIdx which Receiver and Dimension to perturb
    rid = MappedParamsIdx[PerturbParamIdx][0]    # Receiver ID
    dimid = MappedParamsIdx[PerturbParamIdx][1]  # Dimension ID
    # Add perturbation to parameter
    perturb = ((PerturbSTD*ReceiverBoundsWidth.iloc[rid][dimid])*np.random.normal(loc=0))  # Gaussian distributed perturbation
    # nu = np.random.random()
    # gamma = PerturbSTD*(T/T0)*np.tan(np.pi*(nu-0.5))
    # bandwidth = ReceiverBoundsWidth.iloc[rid][dimid]
    # perturb = gamma*bandwidth
    newparam = R.iloc[rid][dimid] + perturb
    # print(perturb)
    # Checks that perturbed parameter lies within the bounds
    isinbound = (newparam >= min(ReceiverBounds.iloc[rid][dimid])) & (newparam <= max(ReceiverBounds.iloc[rid][dimid]))
    # updates receiver parameter (only if new paramater fall within parameter bounds)
    R_prime = pd.DataFrame.copy(R)
    if isinbound == True:
        R_prime.iloc[rid][dimid] = newparam
    return R_prime, isinbound, PerturbParamIdx


def optimizeArray(ReceiverBounds, nReceivers, AnnealingSchedule, S, Rpairs, V, NoiseVariance):
    # start clock
    start = time.time()

    # Defines width of parameters bounds
    ReceiverBoundsWidth = getReceiverBoundsWidth(ReceiverBounds)

    # initialization of variables
    Cost = pd.DataFrame({'T': [], 'cost': []})
    acceptRateChanges = pd.DataFrame({'T': [], 'acceptRate': []})
    acceptRate = 1
    PerturbParamIdx = -1
    Tidx = 0  # temperature step index
    LoopStopFlag = 0
    Rchanges = []
    while LoopStopFlag == 0:  # Temperature loop. Keeps iterating until acceptance rate is too low
        # for nnn in range(1):
        # First iteration
        if (Tidx == 0):
            R = initializeReceivers(nReceivers, ReceiverBounds)  # random initialization of receivers locations (whithin the bounds)
            E_m = getCost(R, S, Rpairs, V, NoiseVariance)        # Calculates max RMS uncertainty
            T = AnnealingSchedule['Start']                       # initial temperature
            tmp1 = pd.DataFrame({'T': [T], 'cost': [E_m]})
            Cost = pd.DataFrame.append(Cost, tmp1, ignore_index=True)
            #Rchanges = R.as_matrix()                                         # Keeps track of model paraneters at each iteration
            Rchanges = R.to_numpy()                                         # Keeps track of model paraneters at each iteration
            MappedParamsIdx = getParamsLinearMapping(R)          # linear list of each elements to optimize (for the perturnation phase)
            #plotArrayUncertainties(R, S, Uncertainties)

        # Checks that starting temperature is not set to low
        if (Tidx == 1):
            if (acceptRate < AnnealingSchedule['StartAcceptanceRate']):
                raise ValueError(['The acceptance rate during the melting phase is too low.Please adjust starting temperature (' + str(acceptRate) + ')'])
            else:
                print('Melting temperature valid (' + str(acceptRate) + ').')

        # Perturb paraneters
        nAccepted = 0  # keeps track of accepted perturbations
        for j in range(AnnealingSchedule['nPerturb']):  # perturbations of parameters for temperature T
            # Perturb one receiver parameter
            PerturbParamIdx += 1  # increment to next parameter in line
            R_prime, isinbound, PerturbParamIdx = perturbReceivers(R, PerturbParamIdx, MappedParamsIdx, ReceiverBounds, ReceiverBoundsWidth, AnnealingSchedule['PerturbSTD'], AnnealingSchedule['Start'], T)
            # Acceptance tests
            acceptedFlag = []
            if isinbound:
                E_mprime = getCost(R_prime, S, Rpairs, V, NoiseVariance)       # Calculates max RMS uncertainty
                deltaE = E_mprime-E_m
                if (deltaE <= 0):   # accept change
                    acceptedFlag = True
                elif (deltaE > 0):
                    psy = np.random.uniform()
                    P = np.exp(-deltaE/T)
                    if (psy <= P):  # accept change
                        acceptedFlag = True
                    else:           # reject change
                        acceptedFlag = False
            else:
                acceptedFlag = False
            # Updates parmaters for accepted changes
            if acceptedFlag:  # accepted change
                nAccepted += 1
                # update paramters
                R = pd.DataFrame.copy(R_prime)
                E_m = E_mprime
                del R_prime, E_mprime   # delete variables
                # saves cost and model parameters
                #tmp1 = pd.DataFrame({'T': [T], 'cost': [E_m]})
                #Cost = pd.DataFrame.append(Cost, tmp1, ignore_index=True)
                #Rchanges = np.dstack((Rchanges, R.as_matrix()))
            elif acceptedFlag is not False:  # sanity check
                raise ValueError(['Perturbation did go throught the acceptance test. Check the code for errors!'])
            # saves cost and model parameters
            tmp1 = pd.DataFrame({'T': [T], 'cost': [E_m]})
            Cost = pd.DataFrame.append(Cost, tmp1, ignore_index=True)
            #Rchanges = np.dstack((Rchanges, R.as_matrix()))
            Rchanges = np.dstack((Rchanges, R.to_numpy()))

        # Calculates acceptance rate for that temperature value
        acceptRate = nAccepted / AnnealingSchedule['nPerturb']
        tmp2 = pd.DataFrame({'T': [T], 'acceptRate': [acceptRate]})
        acceptRateChanges = acceptRateChanges.append(tmp2, ignore_index=True)
        print('Temperature: %.3f - Acceptance rate: %.2f - Cost: %.2f' % (T, acceptRate, E_m))

        # Stopping conditions
        if (acceptRate < AnnealingSchedule['StopAcceptanceRate']):
            LoopStopFlag = 1
            print('Optimization complete (acceptance rate threshold reached)')
        if (E_m <= AnnealingSchedule['StopCost']):
            LoopStopFlag = 1
            print('Optimization complete (cost objective reached)')

        # Update temperature
        T = T * AnnealingSchedule['ReducFactor']  # decrease temperature
        Tidx += 1  # next temperature step

    end = time.time()
    elapsedTime = end - start
    return R, Rchanges, acceptRateChanges, Cost, elapsedTime

def plotOptimizationResults(outdir, nReceivers, Rchanges, Cost, acceptRateChanges, R, iteridx=1):

    # plot Parameters evolution with Temperature
    f1 = plt.figure(1)
    for Ridx in range(nReceivers):
        plt.subplot(nReceivers,1, Ridx+1)
        plt.plot(Rchanges[Ridx,0,:], label='X(m)', color='black')
        plt.plot(Rchanges[Ridx,1,:], label='Y(m)', color='red')
        plt.plot(Rchanges[Ridx,2,:], label='Z(m)', color='green')
        plt.grid(True)
        plt.ylabel('H' + str(Ridx+1) )
        if Ridx == nReceivers -1:
            plt.xlabel('Temperature step')
        #if Ridx == 0:
        #    plt.legend(loc="best", labels=['X(m)','Y(m)','Z(m)'], bbox_to_anchor=(0.5,-0.1))
    f1.savefig(os.path.join(outdir, 'ReceiversPositionVsTemperature' + '_iteration-' + str(iteridx+1) + '.png'), bbox_inches='tight')

    # plot cost evolution with Temperature
    f2 = plt.figure(2)
    plt.plot(Cost['cost'], color = 'black')
    plt.grid(True)
    plt.xlabel('Temperature step')
    plt.ylabel('Cost')
    f2.savefig(os.path.join(outdir, 'CostVsTemperature' + '_iteration-' + str(iteridx+1) + '.png'), bbox_inches='tight')


    # plot acceptance rate with Temperature
    f3 = plt.figure(3)
    plt.semilogx(acceptRateChanges['T'],acceptRateChanges['acceptRate'], color = 'black')
    plt.grid(True)
    plt.xlabel('Temperature')
    plt.ylabel('Acceptance rate')
    plt.semilogx
    f3.savefig(os.path.join(outdir, 'AcceptanceRateVsTemperature' + '_iteration-' + str(iteridx+1) + '.png'), bbox_inches='tight')


    # plot Final receivers positions
    f4 = plt.figure(4)
    ax1 = f4.add_subplot(111, projection='3d')
    # Receivers
    ax1.scatter(R['x'], R['y'], R['z'], s=30, c='black')
    # Axes labels
    ax1.set_xlabel('X (m)', labelpad=10)
    ax1.set_ylabel('Y (m)', labelpad=10)
    ax1.set_zlabel('Z (m)', labelpad=10)
    plt.show()
    f4.savefig(os.path.join(outdir, 'FinalReceiversPosition' + '_iteration-' + str(iteridx+1) + '.png'), bbox_inches='tight')
