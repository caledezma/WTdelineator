#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:01:13 2018

@author: carlos
"""

import matplotlib.pyplot as plt
import WTdelineator as wav
import wfdb
import numpy as np


dbase = 'staffiii/data'
rec = '052c'
sNum = 1

# When in linux
s, att = wfdb.srdsamp(rec,pbdir=dbase)
annot = wfdb.rdann(rec, 'event', pbdir=dbase)
sName = att['signame']

## When in Windows
#s, att = wfdb.rdsamp(rec,pb_dir=dbase)
#annot = wfdb.rdann(rec, 'event', pb_dir=dbase)
#sName = att['sig_name']

# Ranges to analyse signal
beg = int(np.floor(2**16))
end = int(np.floor(2*2**16))

#%%
fs = att['fs']
sig = s[beg:end,sNum]
N = sig.shape[0]
t = np.arange(0,N/fs,1/fs)

# Wavelet Transform delineation
Pwav, QRS, Twav = wav.signalDelineation(sig,fs)

# Calculate biomarkers
QRSd = QRS[:,-1] - QRS[:,0]

Tind = np.nonzero(Twav[:,0])
QT = Twav[Tind,-1] - QRS[Tind,0]
Td = Twav[Tind,-1] - Twav[Tind,0]

Pind = np.nonzero(Pwav[:,0])
Pd = Pwav[Pind,-1] - Pwav[Pind,0]



plt.figure()
plt.plot(t,sig,label=sName[sNum])
plt.plot(t[QRS[:,0]],sig[QRS[:,0]],'*r',label='QRSon', markersize=15)
plt.plot(t[QRS[:,1]],sig[QRS[:,1]],'*y',label='Q', markersize=15)
plt.plot(t[QRS[:,2]],sig[QRS[:,2]],'*k',label='R', markersize=15)
plt.plot(t[QRS[:,3]],sig[QRS[:,3]],'*m',label='S', markersize=15)
plt.plot(t[QRS[:,4]],sig[QRS[:,4]],'*g',label='QRSend', markersize=15)
plt.plot(t[Twav[:,0]],sig[Twav[:,0]],'^r',label='Ton', markersize=10)
plt.plot(t[Twav[:,1]],sig[Twav[:,1]],'^k',label='T1', markersize=10)
plt.plot(t[Twav[:,2]],sig[Twav[:,2]],'^m',label='T2', markersize=10)
plt.plot(t[Twav[:,3]],sig[Twav[:,3]],'^g',label='Tend', markersize=10)
#plt.plot(t[Pwav[:,0]],sig[Pwav[:,0]],'or',label='Pon', markersize=10)
#plt.plot(t[Pwav[:,1]],sig[Pwav[:,1]],'ok',label='P1', markersize=10)
#plt.plot(t[Pwav[:,2]],sig[Pwav[:,2]],'om',label='P2', markersize=10)
#plt.plot(t[Pwav[:,3]],sig[Pwav[:,3]],'og',label='Pend', markersize=10)
plt.title('Delineator output')
plt.xlabel('Time (s)')
plt.ylabel('ECG (mV)')
plt.legend()