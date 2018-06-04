#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:01:44 2018

@author: carlos
"""

#%%
import matplotlib.pyplot as plt
import h5py
import wfdb
import numpy as np
import csv
import ECGprocessing as ecg

dbase = 'staffiii/data'
fName = 'delinResults.h5'
tests = ['BR', 'BC1', 'BC2', 'BI1', 'BI2', 'BI3', 'BI4', 'BI5', 'PC1', 'PC2', 'PR1', 'PR2']
leadNames = ['V1','V2','V3','V4','V5','V6','DI','DII','DIII','aVR','aVL','aVF']
h5attr = [['Annotation names'],['Pon,P,Pend, QRSon,R,QRSend,Ton,Tp,Tend']]

# Read annotations from the purpose made file
annot = []
with open('annotations.csv') as csvfile:
    ann = csv.reader(csvfile,delimiter=',',quotechar='"')
    for row in ann:
        annot += [row]
        
patientNum = np.arange(len(annot)-1) + 1

# Initialise the solution, only needed if not using h5py
           #BR, BC1, BC2, BI1, BI2, BI3, BI4, BI5, PC1, PC2, PR1, PR2
#delinDB =  [[],  [],   [],  [],  [],  [], [],  [],  [],  [],  [],  []]

#%%
# Open file to save the results
outFile = h5py.File(fName,'w')

# Process one patient at a time
for patient in range(1,len(annot)):
    print('Processing patient', patient)
    # Process BR, BC1 and BC2
    for measurement in range(0,3):
        rec = annot[patient][measurement] # Find the name of the record
        if rec != '': # Check that the record exists
            try:
                s, att = wfdb.srdsamp(rec.zfill(4),pbdir=dbase) # Read from Physionet
                # Calculate augmented limb leads and append them to the signals
                aVR, aVL, aVF = ecg.augmentedLimbs(s[:,-3], s[:,-2])
                s = np.concatenate((s, aVR, aVL, aVF), axis=1) 
                
                # Delineate all the ECG leads using the WT and fusion techniques
                ECGdelin = ecg.delineateMultiLeadECG(s,att['fs'])
                
                # Create subgroup for the patient and save all the leads
                grp = outFile.create_group(tests[measurement] + '/' + str(patient).zfill(3))
                for idx, ECG in enumerate(ECGdelin):
                    dsetName =  leadNames[idx]
                    dset = grp.create_dataset(dsetName,ECG.shape,ECG.dtype)
                    dset[...] = ECG
                    #dset.attrs.create(h5attr[0],h5attr[1])                   
                    
                #delinDB[measurement] += [ECGdelin]
            except ValueError:
                print('There was an error when reading file', rec+',', 'file skipped')
                #delinDB[measurement] += ['Error reading record ' + rec]
                
#        else: # If the record doesn't exist, add an empty list
#            delinDB[measurement] += [[]]
            
    # Process BI1 - BI5
    k = 3
    for measurement in range(3,12,2):
        rec = annot[patient][measurement] # Find the name of the record
        if rec != '': # Check that the record exists
            try:
                s, att = wfdb.srdsamp(rec.zfill(4),pbdir=dbase) # Read from Physionet
                
                # Read the annotations
                a1,a2,_ = annot[patient][measurement+1].split(';')
                a1 = int(int(a1)*att['fs']) # Start of balloon inflation
                a2 = int(int(a2)*att['fs']) # End of balloon inflation
                
                # Only keep the ischemic part of the signal
                s = s[a1:a1+a2,:]
                
                # Calculate augmented limb leads and append them to the signals
                aVR, aVL, aVF = ecg.augmentedLimbs(s[:,-3], s[:,-2])
                s = np.concatenate((s, aVR, aVL, aVF), axis=1) 
                
                # Delineate all the ECG leads using the WT and fusion techniques
                ECGdelin = ecg.delineateMultiLeadECG(s,att['fs'])
                
                # Create subgroup for the patient and save all the leads
                grp = outFile.create_group(tests[k] + '/' + str(patient).zfill(3))
                for idx, ECG in enumerate(ECGdelin):
                    dsetName = leadNames[idx]
                    dset = grp.create_dataset(dsetName,ECG.shape,ECG.dtype)
                    dset[...] = ECG
                    #dset.attrs.create(h5attr[0],h5attr[1])   
                    
#                delinDB[k] += [ECGdelin]
            except ValueError:
                print('There was an error when reading file', rec)
                print('File skipped')
                
#                delinDB[measurement] += ['Error reading record ' + rec]            
#        else: # If the record doesn't exist, add an empty list
#            delinDB[k] += [[]]
        k += 1
            
                    
    # Process PC1, PC2, PR1, PR2
    k = 8
    for measurement in range(13,17):
        rec = annot[patient][measurement] # Find the name of the record
        if rec != '': # Check that the record exists
            try:
                s, att = wfdb.srdsamp(rec.zfill(4),pbdir=dbase) # Read from Physionet
                # Calculate augmented limb leads and append them to the signals
                aVR, aVL, aVF = ecg.augmentedLimbs(s[:,-3], s[:,-2])
                s = np.concatenate((s, aVR, aVL, aVF), axis=1) 
                
                # Delineate all the ECG leads using the WT and fusion techniques
                ECGdelin = ecg.delineateMultiLeadECG(s,att['fs'])
                
                # Create subgroup for the patient and save all the leads
                grp = outFile.create_group(tests[k] + '/' + str(patient).zfill(3))
                for idx, ECG in enumerate(ECGdelin):
                    dsetName = leadNames[idx]
                    dset = grp.create_dataset(dsetName,ECG.shape,ECG.dtype)
                    dset[...] = ECG
                    #dset.attrs.create(h5attr[0],h5attr[1])   
#                delinDB[k] += [ECGdelin]
            except ValueError:
                print('There was an error when reading file', rec)
                print('File skipped')
#                delinDB[measurement] += ['Error reading record ' + rec]
#        else: # If the record doesn't exist, add an empty list
#            delinDB[k] += [[]]
        k += 1


# Close file
outFile.close()
        
