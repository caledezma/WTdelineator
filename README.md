# WTdelineator
Wavelet-based ECG delineator library implemented in python

The library required to perform ECG delineation is provided, along with instructions of use, in WTdelineator.py. This implementation is based on the work: Martínez, Juan Pablo, et al. "A wavelet-based ECG delineator: evaluation on standard databases." IEEE transactions on biomedical engineering 51.4 (2004): 570-581.

Two examples are provided, one performs delineation on a single signal (delineateSignal.py) of the STAFFIII database (https://physionet.org/physiobank/database/staffiii/) and the other on the whole database (delineateDatabase.py). They both require the pyhon PhysioToolkit WFDB python package (https://github.com/MIT-LCP/wfdb-python). The second example requires the annotations.csv file, which is a summarised version of the annotations provided in the STAFFIII files.

All the functions contained in this repository were developed in the Multiscale Cardiovascular Engineering Group at University College London (MUSE-UCL) by Carlos Ledezma.

This work is protected by a Creative Commons Attribution-ShareAlike 4.0 International license (https://creativecommons.org/licenses/by-sa/4.0/)
