#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: 30/04/2018

This library contains generic functions for ECG signal processing. It was 
developed to be applied for the delineation of signals from the STAFFIII
database (https://physionet.org/physiobank/database/staffiii/). 

The delineateMultiLeadECG() requires the WTdelineator.py library.

Details about the fusion technique used in the detectionVoteFusion() function
can be found in: "Data fusion for QRS complex detection in multi-lead 
electrocardiogram recordings" by Ledezma et al. (2015).

All the functions contained in this library were developed in the Multiscale
Cardiovascular Engineering Group at University College London (MUSE-UCL)
by Carlos Ledezma.

This work is protected by a Creative Commons Attribution-ShareAlike 4.0 
International license (https://creativecommons.org/licenses/by-sa/4.0/)
"""

import numpy as np

def augmentedLimbs(DI,DII):
    '''
    aVR, aVL, aVF = augmentedLimbs(DI,DII)
    
    Calculate the ECG augmented limb leads using leads DI and DII.
    
    Input:
        DI (numpy array): contains the DI lead.
        
        DII (numpy array): contains the DII lead.
        
    Output:
        aVR (numpy array): contains the aVR lead calculated as:
            aVR = -1/2 * (DI + DII)
        
        aVL (numpy array): contains the aVL lead calculated as:
            aVL = DI - 1/2 * DII
            
        aVF (numpy array): contains the aVF lead calculated as:
            aVF = -1/2 * DI + DII
    '''
    aVR = np.atleast_2d(-1/2 * (DI + DII)).T
    aVL = np.atleast_2d(DI - 1/2 * DII).T
    aVF = np.atleast_2d(-1/2 * DI + DII).T
    
    return aVR, aVL, aVF

def detectionVoteFusion(det,win):
    '''
    detFusion = detectionVoteFusion(det,win)
    
    Performs vote fusion with the detections provided. For this to work the 
    detection window has to be consistent with the detections provided. This
    means that if the detections are in sample number, the window has to be 
    given in number of samples. Alternatively, if the samples are given in 
    seconds, the window must be given in seconds. And so on...
    
    For each detection vector (ECG lead) the function compares each detection
    with those found in the other detection vectors. If at least half of the
    leads contain a detection close to the one being analysed (as defined by 
    the window), then the detection is validated.
    
    Input:
        det (list): contains the detections made on the different leads. Each 
        lead must be provided as a list where each entry is a detection mark.
        
        win (float): window that defines when two detections in different leads
        are considered the same. This input must be consistent with the values
        given in the detection vectors
        
    Output:
        detFusion (list): each list in this variables corresponds to one list
        provided in det. The marks provided in detFusion are only those that 
        were validated for each lead.
    '''
    
    # Initialise the solution list
    detFusion = []
    
    for lead in det: # Verify one lead at a time
        detFusion += [[]] # Create a new list for the current lead
        
        if len(lead) > 0:
            for cand in lead: # Iterate over alll the detections in the lead
                vote = 0 # Initialise voting variable for each candidate
                for comp in det: # Compare with all the leads
                    # Signal if the detection appears in another lead
                    if len(comp) > 0:
                        if np.min(np.abs(np.array(comp) - cand)) < win: 
                            vote += 1
                                
                # If the detection is confirmed in more than half of the channels
                # the index is saved as a confirmed detection in the lead
                if vote/len(det) >= 0.5:
                    detFusion[-1] += [cand]

    return detFusion

def delineateMultiLeadECG(sig,fs):
    '''
    ECGdelin = delineateMultiLeadECG(sig,fs)
    
    Delineate multi-lead ECG using WT transform and detection fusion. This 
    function requires the WTdelineator library. 
    
    The algorithm performs QRS detection on each lead using the wavelet 
    transform. Then, it performs data fusion on the marks to keep only 
    the reliable beat detections. Finally, the wavelet-based delineation 
    process continues using only the validated R-peak marks.
    
    Input:
        sig (numpy array): contains all the ECG leads to be analysed. Each 
        column of the array is assumed to be a lead and each row a sample.
        
        fs (float): sampling frequency, in Hz, at which the ECG was acquired.
        
    Output:
        ECGdelin (list): ECG delneation result. Each entry is the delineation
        result (numpy array) for an ECG lead, which correspond to those 
        provided in sig. Each row of a delineation result correspond to a 
        found and validated beat and it's of the form:
            
            ECGdelin[k][i,:] = [Pon, P, Pend, QRSon, R, QRSend, Ton, T, Tend]    
    '''
    
    import WTdelineator as wt
    
    # The WT transform should only use 2**16 samples at a time:
    beg = 0
    end = 2**16
    last = False
    
    # Initialise the lists that will contain the results
    ECGdelin = []
    
    for i in range(sig.shape[1]):
        ECGdelin += [np.zeros((1,9),dtype=int)]
    
    while not last: # Continue delineating until the end of the signal
        # Initialise temporary arrays
        R = []
        QRSon = []
        QRSend = []
        n = []
        Tp = []
        Ton = []
        Tend = []
        Pp = []
        Pon = []
        Pend = []
        
        # If the end of the signal was reached in the last iteration, signal
        # that the last segment is being processed.
        if end >= sig.shape[0]:
            end = sig.shape[0]
            last = True
            
        # Define the WT filters
        thisSig = sig[beg:end,:]
        N = thisSig.shape[0] # Number of samples in the signal
        WTfilters = wt.waveletFilters(N,fs) # Create the filters to apply the algorithme-a-trous
        w = []
        
        for lead in range(sig.shape[1]): # Find QRS complexes in each lead
            w += [wt.waveletDecomp(thisSig[:,lead], WTfilters)] # Save the decomposition of each lead
            Rb, nb = wt.Rdetection(w[-1],fs)
            R += [Rb]
            n += [nb]
        
        # Perform fusion to keep only valid QRS complexes
        win = int(np.floor((40/1000) * fs))
        R = detectionVoteFusion(R, win)
        
        # Complete the delineation with the reliable beats
        for lead in range(len(R)):            
            _, _, Onb, Endb = wt.QRSdelineation(R[lead],w[lead],n[lead],fs)
            T1,T2,TonB,TendB = wt.Tdelineation(Endb,w[lead],fs) # Detect and delineate the T wave
            P1,P2,PonB,PendB = wt.Pdelineation(Onb,w[lead],fs) # Detect and delineate the P wave
            
            # Only keep the largest of the T wave peaks
            TpB = []
            for idx, Tcand in enumerate(T1): 
                if Tcand != 0:
                    if (np.abs(thisSig[Tcand,lead]) > np.abs(thisSig[T2[idx],lead])) or (T2[idx] == 0):
                        TpB += [Tcand]
                    else:
                        TpB += [T2[idx]]
                else:
                    TpB += [Tcand]
                    
            # Only keep the largest of the P wave peaks
            PpB = []
            for idx, Pcand in enumerate(P1): 
                if Pcand != 0:
                    if (np.abs(thisSig[Pcand,lead]) > np.abs(thisSig[P2[idx],lead])) or (P2[idx] == 0):
                        PpB += [Pcand]
                    else:
                        PpB += [P2[idx]]
                else:
                    PpB += [Tcand]
            
            # Save the detections of this lead 
            QRSon += [Onb]
            QRSend += [Endb]
            Tp += [TpB]
            Ton += [TonB]
            Tend += [TendB]
            Pp += [PpB]
            Pon += [PonB]
            Pend += [PendB]
        
        # Convert everything to a numpy array and append the results to the
        # end of the variables
        for i in range(sig.shape[1]):
            QRSonB = np.array(QRSon[i],ndmin=2,dtype=int)
            QRSonB += beg * (QRSonB>0)
            
            Rb = np.array(R[i],ndmin=2,dtype=int)
            Rb += beg * (Rb>0)
            
            QRSendB = np.array(QRSend[i],ndmin=2,dtype=int)
            QRSendB += beg * (QRSendB>0)
            
            TpB = np.array(Tp[i],ndmin=2,dtype=int)
            TpB += beg * (TpB>0)
            
            TonB = np.array(Ton[i],ndmin=2,dtype=int)
            TonB += beg * (TonB>0)
            
            TendB = np.array(Tend[i],ndmin=2,dtype=int)
            TendB += beg * (TendB>0)
            
            PpB = np.array(Pp[i],ndmin=2,dtype=int)
            PpB += beg * (PpB>0)
            
            PonB = np.array(Pon[i],ndmin=2,dtype=int)
            PonB += beg * (PonB>0)
            
            PendB = np.array(Pend[i],ndmin=2,dtype=int)
            PendB += beg * (PendB>0)
            
            ECGdelinB = np.concatenate((PonB,PpB,PendB,\
                                        QRSonB,Rb,QRSendB,\
                                        TonB,TpB,TendB),axis=0).T
            ECGdelin[i] = np.concatenate((ECGdelin[i],ECGdelinB), axis=0)
        
        beg = end
        end += 2**16
        
    for i in range(sig.shape[1]): # Eliminate the initialisation line
        ECGdelin[i] = np.delete(ECGdelin[i],0,axis=0)
        
    return ECGdelin


            