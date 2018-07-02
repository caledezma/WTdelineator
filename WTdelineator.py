#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: 30/04/2018

This library contains all the functions required to perform wavelet-based
ECG delineation. The full delineation can be made with the function:
    
    P, QRS, T = signalDelineation(sig,fs):
        
Ideally, the function should be used with a maximum of 2^16 samples. This 
guarantees that the thresholds are properly updated and avoids errors due 
to transient, high magnitude, noise.

The delineation algorithm is explained in "A wavelet-based ECG delineator: 
evaluation on standard databases" by Martinez et al. (2003). The maximum modulus 
lines are found as explained in the paper: "Detection of ECG characteristic 
points using wavelet transforms." by Li et al. (1995). Users are invited to 
consult these papers in case there are doubts about the algorithms.

All the functions contained in this library were developed in the Multiscale
Cardiovascular Engineering Group at University College London (MUSE-UCL)
by Carlos Ledezma.

This work is protected by a Creative Commons Attribution-ShareAlike 4.0 
International license (https://creativecommons.org/licenses/by-sa/4.0/)
"""
import numpy as np
from scipy.signal import resample

def waveletH(w):
    '''    
    H = waveletH(w)
    
    Constructs the low-pass filters required for the wavelet-based ECG delineator
    at a sampling frequency of 250 Hz.
    
    Input:
        w (numpy array): contains the frequency points, in radians, that 
        will be used to construct the filter. w must be between 0 and 2pi.
    
    Output:
        H (numpy array): contains the function H(w) = exp(1j*w/2) * cos(w/2)**3. 
    '''
    return np.exp(1j*w/2) * np.cos(w/2) ** 3 
    
    
def waveletG(w):
    '''    
    G = waveletG(w)
    
    Constructs the high-pass filters required for the wavelet-based ECG delineator
    at a sampling frequency of 250 Hz.
    
    Input:
        w (numpy array): contains the frequency points, in radians, that 
        will be used to construct the filter. w must be between 0 and 2pi.
    
    Output:
        G (numpy array): contains the function H(w) = exp(1j*w/2) * cos(w/2)**3. 
    '''
    
    return 4j*np.exp(1j*w/2)*np.sin(w/2)

def waveletFilters(N,fs):
    '''    
    Q = waveletFilters(N,fs)
    
    Creates the filters required to make the wavelet decomposition using the
    algorithme-a-trous. This routine first creates the filters at 250 Hz and 
    resamples them to the required sampling frequency.
    
    Inputs:
        N (int): the number of samples of the signal that will be decomposed.
        
        fs (float): the sampling frequency of the signal that will be decomposed.
        
    Output:
        Q (list): contains five numpy arrays [Q1, Q2, Q3, Q4, Q5] that are the 
        five filters required to make the wavelet decomposition.
    '''
    
    # M is the number of samples at 250 Hz that will produce N samples after 
    # re-sampling the filters    
    M =  N* 250/fs
    w = np.arange(0,2*np.pi, 2*np.pi/M) # Frequency axis in radians
    
    # Construct the filters at 250 Hz as specified in the paper
    Q = [waveletG(w)]    
    for k in range(2,6):
        G = waveletH(w)
        for l in range(1,k-1):
            G *= waveletH(2**l * w)
        Q += [waveletG(2 ** (k-1) * w) * G]
        
    # Resample the filters from 250 Hz to the desired sampling frequency
    for i in range(len(Q)):
        Q[i] = np.fft.fft(resample(np.fft.ifft(Q[i]),N))
    
    return Q    
    
def waveletDecomp(sig,Q):
    '''
    w = waveletDecomp(sig,Q)
    
    Performs the wavelet decomposition of a signal using the algorithme-a-trous. 
    
    Inputs:
        sig (numpy array): contains the signal to be decomposed.
        
        Q (list): contains the filters (numpy arrays) that will decompose the
        signal. It is recommended that Q is generated using the waveletFilters 
        function provided in this library.
        
    Output:
        
        w (list): numpy arrays [w1, w2, w3, w4, w5] containing the wavelet
        decomposition of the signal at scales 2^1..2^5. 
    '''
       
    w = []
    
    # Apply the filters in the frequency domain and return the result in the time domain
    for q in Q:
        w += [np.real(np.fft.ifft(np.fft.fft(sig) * q))]
        
    return w

def findMaxModulus(sig, start, end, thresh):
    '''    
    n = findMaxModulus(sig, start, end, thresh)
    
    Finds the maximum modulus lines (MML) in a signal, as explained by Li et al. 
    with the improvements proposed by Martinez et al. 
    
    Inputs:
        sig (numpy array): contains the signal where the maximum modulus lines
        will be looked for
        
        start (int): sample number specifying where to start looking for MML 
        in sig.
        
        end (int): sample number specifying where to finish looking for MML 
        in sig.
        
        thresh (float): threshold above which a local maximum is considered a 
        MML.
        
    Output:
        n (list): containing the sample numbers (int) of the signal where MML 
        were found.
    
    '''
    n = []
    lookingForMax = False
    
    # Verify that the end point is not after the end of the signal
    if end >= sig.shape[0]:
        end = sig.shape[0]-2
        
    if start < 0:
        start = 0        
    
    # Look for local maxima that are above the specified threshold
    # Since MML can be positive or negative, local maxima are assessed on the
    # absolute value of the signal.
    
    for i in range(start,end):
        # Only look for a maximum if the signal is above the threshold and 
        # we are in a rising slope
        if abs(sig[i]) >= thresh and abs(sig[i-1]) < abs(sig[i]) and not(lookingForMax):
            lookingForMax = True
        elif lookingForMax:
            # If the conditions are met, save the position of the MML
            if abs(sig[i]) >= abs(sig[i-1]) and abs(sig[i]) >= abs(sig[i+1]):
                n += [i]
                lookingForMax = False
    return n

def verifyMaxModulus(sig,MML,win,thresh):
    '''    
    n = verifyMaxModulus(sig,MML,win,thresh)
    
    Verifies that a set of maximum modulus lines found in a given signal occur
    again in the signal provided.
    
    Inputs:
        sig (numpy array): contains the signal where the verification of the 
        MML will be made.
        
        MML (list): contains the MML (int) that will be verified.
        
        win (int): specifying how far, from the positions provided in MML, to 
        verify the maximum modulus line.
        
        thresh (float): the value of the threshold above which a maximum is 
        considered significant.
        
    Output:
        n (list): contains the sample numbers (int) of the signal where the 
        maximum modulus lines were confirmed.
    '''
            
    n = []
    for m in MML:
        cand = findMaxModulus(sig,m,m+win,thresh)
        if len(cand) > 0: # Verify that a MML exists in the vicinity of m
            if len(cand) == 1: # If there is only one modulus close by then the search is over
                n += cand
            else:     
                # If there are several maximum modulus lines, we find the closest to n
                closest = cand[np.argmin(abs(np.array(cand)-m))]
                # If the largest MML is smaller than 1.2 times the closest, 
                #then select the closest (distinguish between positive and negative)
                if sig[m] > 0:
                    if max(sig[cand]) < 1.2 * sig[closest]:
                        n += [closest]
                    else: 
                        n += [cand[np.argmax(sig[cand])]]
                else:
                    if min(sig[cand]) > 1.2 * sig[closest]:
                        n += [closest]
                    else: 
                        n += [cand[np.argmin(sig[cand])]]
    return n

def findWaveLimits(w,beg,thresh,on, win):
    '''        
    pos = findWaveLimits(w,beg,thresh,on)
    
    Finds the beginning or the end a ECG wave. 
    
    Inputs:
        w (numpy array): the signal where the beginning or end of the wave
        will be identified.
        
        beg (int): the sample number where to start the search.
        
        thresh (float): the threshold under which the limit of the wave is said
        to be found.
        
        on (bool): specifies if looking for the onset (True) or the end (False)
        of the ECG wave.
        
        win (int): maximum number of samples after, or before, the starting 
        position to look for the wave limit.
    
    Output:
        pos (list): contains a single element with the sample number where the
        wave limit was found.
        
    '''
    # The search goes backward if looking for the onset or forward if looking
    # for the end of the ECG wave
    if on:
        update = -1
    else:
        update = 1
    
    # The search begins one sample before (or after) the given point
    searchPos = beg + update        
    
    while abs(searchPos-beg) < win and searchPos >= 1 and searchPos <= w.shape[0]-2: # Loop only within the search limits
        # If the conditions for wave limit are met, return the position
        if (abs(w[searchPos]) < thresh) or (abs(w[searchPos]) > abs(w[searchPos+1]) and abs(w[searchPos]) > abs(w[searchPos-1])):
            return [searchPos]
        else: # Otherwise, look at the next sample
            searchPos += update
            
    return [searchPos] # If the limit is reached, return the limit

def Rdetection(w,fs):
    '''    
    R, n = Rdetection(w,fs)
    
    Find the position of the R peaks of an ECG signal using its wavelet 
    decomposition.
    
    Inputs:
        w (list): contains numpy arrays [w1, w2, w3, w4] which are scales 
        2^1..2^4 of the wavelet decomposition of an ECG signal. This 
        decomposition should be obtained using the waveletDecomp() function
        provided in this library.
        
        fs (float): the sampling frequency of the signal.
        
    Outputs:
        R (list): the sample numbers where the R peaks were detected.
        
        n (list): contains four lists [n1,n2,n3,n4] that contain the sample
        numbers where the maximum modulus lines were found at each scale.
    '''
    
    eps = []
    
    # Window defining when two maximum modulus lines are close enough
    win = int(np.floor((40/1000) * fs))
    
    # Thresholds that define a significant maximum modulus line
    for sig in w:
        eps += [np.sqrt(np.mean(sig**2))]
        
    eps[3] /= 2
    
    # Find maximus modulus lines at the 4th scale
    n4 = findMaxModulus(w[3],1,w[3].shape[0]-1,eps[3])   
    
    # Use n4 to find maximum modulus lines in 3rd scale
    n3 = verifyMaxModulus(w[2],n4,win,eps[2])
    
    # Use n3 to find maximum modulus lines in 2nd scale
    n2 = verifyMaxModulus(w[1],n3,win,eps[1])
    
    # Use n2 to find maximum modulus lines in 1st scale
    n1 = verifyMaxModulus(w[0],n2,win,eps[0])
        
    R = []
    # Detect R peaks by finding zero crossings between the remaining maximus 
    # modulus lines. The conditions ensure that isolated or redundant lines are ignored
    for i in range(1,len(n1)):
        #   MMLs are close enough                            MMLs have opposite signs       
        if (n1[i]-n1[i-1] < win) and ((w[0][n1[i]]>0 and w[0][n1[i-1]]<0) or (w[0][n1[i]]<0 and w[0][n1[i-1]]>0)):
            if w[0][n1[i-1]:n1[i]].size != 0: # Safety
                R += [np.argmin(abs(w[0][n1[i-1]:n1[i]])) + n1[i-1]]
                
    # Filter redundant R peaks
    win = int(np.floor(50/1000*fs))
    i = 1
    while i < len(R):
        if np.abs(R[i] - R[i-1]) < win:
            del R[i]
        else:
            i += 1
            
    return R, [n1,n2,n3,n4]

def QRSdelineation(R,w,n,fs):
    '''    
    Q, S, QRSon, QRSend = QRSdelineation(R,w,n,fs)
    
    Perform QRS complex deliniation.
    
    Inputs:
        R (list): Contains the sample numbers (int) where R peaks have been 
        previously found. This input should be obtained using the Rdetection()
        function provided in this library.
        
        w (list): contains numpy arrays [w1, w2, w3, w4] which are scales 
        2^1..2^4 of the wavelet decomposition of an ECG signal. This 
        decomposition should be obtained using the waveletDecomp() function
        provided in this library.
        
        n (list): contains four lists [n1,n2,n3,n4] that contain the sample
        numbers where maximum modulus lines were found at each scale. This input 
        should be obtained using the Rdetection() function provided in this 
        library.
        
        fs (float): sampling frequency of the signal.
        
    Outputs:
        Q (list): sample numbers (int) where the peak of the Q wave was found.
        The list contains one entry per R peak provided, 0 means that no Q wave
        was found around the corresponding R peak.
        
        S (list): sample numbers (int) where the peak of the S wave was found.
        The list contains one entry per R peak provided, 0 means that no S wave
        was found around the corresponding R peak.
        
        QRSon (list): sample numbers (int) denoting the onset of each QRS complex.
        
        QRSend (list): sample numbers (int) denoting the end of each QRS complex.
    '''
    
    Q = []
    S = []
    QRSon = []
    QRSend = []
    # Window to look for Q or S wave before or after the R peak
    win = int(np.floor((65/1000)*fs)) 
    
    # Loop through all the R peaks provided
    for nqrs in R:
        if nqrs-win >= 0:
            beg = nqrs-win
        else:
            beg = 0
        if nqrs+win < w[1].shape[0]:
            end = nqrs+win
        else:
            end =  w[1].shape[0]
        # Define the thresholds to find the Q and S waves
        gQRSpre = 0.06*np.max(np.abs(w[1][beg:end]))
        gQRSpost = 0.09*np.max(np.abs(w[1][beg:end]))
        
        # Find the maximum modulus lines at scale 2^2 that correspond to npre and npost
        if nqrs < n[1][-1]:
            npost = np.nonzero((n[1]-nqrs)>0)[0][0]
        else:
            npost = len(n[1])-1
        npre = npost - 1
        npost = n[1][npost]
        npre = n[1][npre]
        
        # Find the Q wave or assign 0 if Q wave not found.
        # nfirst is assigned depending on the presence or not of Q
        Qb = findMaxModulus(w[1], npre-win, npre, gQRSpre)
        if len(Qb) > 0:
            Qb = Qb[-1]
            Q += [np.argmin(abs(w[0][Qb:npre])) + Qb]
            nfirst = np.copy(Qb)
        else:
            Q += [0]
            nfirst = np.copy(npre)
        
        # Find the S wave or assign 0 if S wave not found
        # nlast is assigned depending on the presence or not of S
        Sb = findMaxModulus(w[1], npost, npost+win, gQRSpost)
        if len(Sb) > 0:
            Sb = Sb[0]
            S += [np.argmin(abs(w[0][npost:Sb])) + npost]
            nlast = np.copy(Sb)
        else:
            S += [0]
            nlast = np.copy(npost)
            
        # Define the thresholds for onset and end of QRS complex
        if w[1][nfirst] > 0:
            xiQRSon = 0.05 * w[1][nfirst]
        else:
            xiQRSon = 0.07 * w[1][nfirst]
            
        if w[1][nlast] > 0:
            xiQRSend = 0.125 * w[1][nlast]
        else:
            xiQRSend = 0.71 * w[1][nlast]
            
        # Find the onset and end of the QRS complex
        win = int(np.floor(50/1000*fs))
        QRSon += findWaveLimits(w[1],nfirst,xiQRSon,True,win)
        QRSend += findWaveLimits(w[1],nlast,xiQRSend,False,win)
                
    return Q,S,QRSon,QRSend

def Tdelineation(QRSend, w, fs):
    '''    
    T1, T2, Ton, Tend = Tdelineation(QRSend,w,fs)
    
    Perform T wave detection and deliniation.
    
    Inputs:
        QRSend (list): Contains the sample numbers (int) the QRS complexes end. 
        This input should be obtained using the QRSdelineation() function 
        provided in this library.
        
        w (list): contains numpy arrays [w1, w2, w3, w4] which are scales 
        2^1..2^4 of the wavelet decomposition of an ECG signal. This 
        decomposition should be obtained using the waveletDecomp() function
        provided in this library.
        
        fs (float): sampling frequency of the signal.
        
    Outputs:
        T1 (list): sample numbers (int) where the first peak of the T wave was found.
        The list contains one entry per QRSend provided, 0 means that no T wave
        was found close to the corresponding QRS complex.
        
        T2 (list): sample numbers (int) where the second peak of the T wave was found.
        The list contains one entry per QRSend provided, 0 means that no second
        peak was found.
        
        Ton (list): sample numbers (int) denoting the onset of each T wave.
        
        Tend (list): sample numbers (int) denoting the end of each T wave.
    '''
    
    T1 = []
    T2 = []
    Ton = []
    Tend = []
    # Window to look for a T wave after QRSend
    SW = int(np.floor((420/1000) * fs))
    # Threshold that determines if maximum modulus lines (MML) are significant
    epsT = 0.25 * np.sqrt(np.mean(w[3]**2))
    
    for beg in QRSend: # Loop over all the QRS complexes
        if beg + SW < w[3].shape[0]: # Check that there is enough signal left
            
            # Second threshold to verify significant MML
            gammaT = 0.125 * np.max(np.abs(w[3][beg : beg+SW]))
            
            # Candidates to MML as defined by epsT
            cand = findMaxModulus(w[3],beg,beg + SW,epsT)
            
            # There must be at least two
            if len(cand) >= 2:
                significantMax = []
                
                # Check that the candidates are larger than the second threshold
                for pos in cand:
                    if np.abs(w[3][pos]) > gammaT:
                        significantMax += [pos]
                
                # The wave peaks are assigned according to the number of
                # significant MML found (either 2 or 3) as the zero crossings
                # between the MMLs; nfirst and nlast are assigned accordingly
                
                if len(significantMax) == 2:
                    T1 += [np.argmin(abs(w[2][significantMax[0]:significantMax[1]])) + significantMax[0]]
                    T2 += [0]
                    nfirst = significantMax[0]
                    nlast = significantMax[1]
                    waveNotFound = False
                elif len(significantMax) > 2:
                    significantMax = significantMax[:3]
                    T1 += [np.argmin(abs(w[2][significantMax[0]:significantMax[1]])) + significantMax[0]]
                    T2 += [np.argmin(abs(w[2][significantMax[1]:significantMax[2]])) + significantMax[1]]
                    nfirst = significantMax[0]
                    nlast = significantMax[-1]
                    waveNotFound = False
                else:
                    waveNotFound = True
            else:
                waveNotFound = True
            
            # If the wave was found, find the onset and end
            if not waveNotFound:
                xiTon = 0.25 * np.abs(w[3][nfirst])
                xiTend = 0.4 * np.abs(w[3][nlast])
                
                # Window to find onset and end with respect to the position of the peak
                win = int(np.floor(80/1000*fs)) 
                # Find onset and end
                Ton += findWaveLimits(w[3],nfirst,xiTon,True,win)
                Tend += findWaveLimits(w[3],nlast,xiTend,False,win)
                
            else: # No wave found
                T1 += [0]
                T2 += [0]
                Ton += [0]
                Tend += [0]
        else: # No wave found
            T1 += [0]
            T2 += [0]
            Ton += [0]
            Tend += [0]
            
    return T1,T2, Ton, Tend

def Pdelineation(QRSon,w,fs):
    '''    
    P1, P2, Pon, Pend = Pdelineation(QRSend,w,fs)
    
    Perform P wave detection and deliniation.
    
    Inputs:
        QRSon (list): Contains the sample numbers (int) the QRS complexes onset. 
        This input should be obtained using the QRSdelineation() function 
        provided in this library.
        
        w (list): contains numpy arrays [w1, w2, w3, w4] which are scales 
        2^1..2^4 of the wavelet decomposition of an ECG signal. This 
        decomposition should be obtained using the waveletDecomp() function
        provided in this library.
        
        fs (float): sampling frequency of the signal.
        
    Outputs:
        P1 (list): sample numbers (int) where the first peak of the P wave was found.
        The list contains one entry per QRSend provided, 0 means that no P wave
        was found close to the corresponding QRS complex.
        
        P2 (list): sample numbers (int) where the second peak of the P wave was found.
        The list contains one entry per QRSon provided, 0 means that no second
        peak was found.
        
        Pon (list): sample numbers (int) denoting the onset of each P wave.
        
        Pend (list): sample numbers (int) denoting the end of each P wave.
    '''
    
    P1 = []
    P2 = []
    Pon = []
    Pend = []
    # Window to look for a P wave before QRSend
    SW = int(np.floor((200/1000) * fs))
    # Threshold that determines if maximum modulus lines (MML) are significant
    epsP = 0.02 * np.sqrt(np.mean(w[3]**2))
    
    for beg in QRSon: # Loop over all the QRS complexes
        if beg - SW >= 0: # Check that there is enouch signal to perform detection
             # Second threshold to verify significant MML
            gammaP = 0.0125 * np.max(np.abs(w[3][beg-SW : beg]))
            
            # Candidates to MML as defined by epsP
            cand = findMaxModulus(w[3],beg - SW, beg, epsP)
            
            # There must be at least two
            if len(cand) >= 2:
                significantMax = []
                
                # Check that the candidates are larger than the second threshold
                for pos in cand:
                    if np.abs(w[3][pos]) > gammaP:
                        significantMax += [pos]
                 
                # The wave peaks are assigned according to the number of
                # significant MML found (either 2 or 3) as the zero crossings
                # between the MMLs; nfirst and nlast are assigned accordingly
                if len(significantMax) == 2:
                    P1 += [np.argmin(abs(w[2][significantMax[0]:significantMax[1]])) + significantMax[0]]
                    P2 += [0]
                    nfirst = significantMax[0]
                    nlast = significantMax[1]
                    waveNotFound = False
                elif len(significantMax) > 2:
                    significantMax = significantMax[:3]
                    P1 += [np.argmin(abs(w[2][significantMax[0]:significantMax[1]])) + significantMax[0]]
                    P2 += [np.argmin(abs(w[2][significantMax[1]:significantMax[2]])) + significantMax[1]]
                    nfirst = significantMax[0]
                    nlast = significantMax[-1]
                    waveNotFound = False
                else:
                    waveNotFound = True
            else:
                waveNotFound = True
            
            # If the wave was found, find the onset and end
            if not waveNotFound:
                xiPon = 0.5 * np.abs(w[3][nfirst])
                xiPend = 0.9 * np.abs(w[3][nlast])
                
                # Window to find onset and end with respect to the position of the peak
                win = int(np.floor(40/1000*fs))
                
                # Find the onset and end of the P wave
                Pon += findWaveLimits(w[3],nfirst,xiPon,True,win)
                Pend += findWaveLimits(w[3],nlast,xiPend,False,win)
                
            else: # Wave not found
                P1 += [0]
                P2 += [0]
                Pon += [0]
                Pend += [0]
        else: # Wave not found
            P1 += [0]
            P2 += [0]
            Pon += [0]
            Pend += [0]   
    
    return P1, P2, Pon, Pend

def signalDelineation(sig,fs):
    '''
    P, QRS, T = signalDelineation(sig,fs)
    
    Perform wavelet-based ECG delineation.
    
    Inputs:
        sig (numpy array): ECG signal to be delineated.
        fs (float): sampling frequency
        
    Output:
        P (numpy array): each row of this array corresponds to one beat found
        in the ECG signal. Each row has the form P[i,:] = [Pon, P1, P2, Pend],
        where each entry is a sample number containing the following:
            Pon is the location of the onset of the P wave,
            P1 the location of its first peak, 
            P2 the location of its second peak and
            Pend the location of the end of the wave. 
        If all entries of a row are zero, it means that no P wave was found 
        in that beat. If P2 is zero, it means that only one peak was found 
        in that P wave.
        
        QRS (numpy array): each row of this array corresponds to one beat found
        in the ECG signal. Each row has the form QRS[i,:] = [QRSon, Q, R, S, QRSend],
        where each entry is a sample number containing the following:
            QRSon is the location of the onset of the QRS complex, 
            Q is the location of the peak of the Q wave,
            R is the location of the peak of the R wave,
            S is the location of the peak of the S wave,
            QRSend is the location of the end of the QRS complex            
        If the Q or S entries of a row are zero, it means that that particular 
        wave was not found in that beat. 
        
        T (numpy array): each row of this array corresponds to one beat found
        in the ECG signal. Each row has the form T[i,:] = [Ton, T1, T2, Tend],
        where each entry is a sample number containing the following:
            Ton is the location of the onset of the P wave,
            T1 the location of its first peak, 
            T2 the location of its second peak and
            Tend the location of the end of the wave. 
        If all entries of a row are zero, it means that no T wave was found 
        in that beat. If T2 is zero, it means that only one peak was found 
        in that T wave.
    '''
        
    N = sig.shape[0] # Number of samples in the signal
    Q = waveletFilters(N,fs) # Create the filters to apply the algorithme-a-trous

    w= waveletDecomp(sig,Q) # Perform signal decomposition
    
    R, n = Rdetection(w,fs) # Detect the R peaks
    
    Q,S,QRSon,QRSend = QRSdelineation(R,w,n,fs) # Delineate QRS complex
    
    T1,T2,Ton,Tend = Tdelineation(QRSend,w,fs) # Detect and delineate the T wave
    
    P1,P2,Pon,Pend = Pdelineation(QRSon,w,fs) # Detect and delineate the P wave
    
    # Create output arrays
    QRS = np.array([QRSon,Q,R,S,QRSend]).T
    Twav = np.array([Ton,T1,T2,Tend]).T
    Pwav = np.array([Pon,P1,P2,Pend]).T
    
    return Pwav, QRS, Twav
