import numpy as np
from statsmodels.graphics import tsaplots
from scipy.signal import butter,filtfilt,savgol_filter
from scipy.signal import find_peaks

import statsmodels.api as sm
import scipy.io
import scipy
import matplotlib.pyplot as plt
#from fiducial_features_11_points  import get_fiducial_features
import neurokit2 as nk
import numpy as np
from sklearn import preprocessing


def get_fiducial_features(ecg_signal):
    
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=1000)
    signal_cwt, waves_cwt = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=1000,method="cwt",show=True,show_type='all')
    
    l=len(waves_cwt['ECG_Q_Peaks'])
    
    feature_array=[]
    for i in range(l-1):

        vector=[]
        vector.append(waves_cwt['ECG_P_Onsets'][i])
        vector.append(waves_cwt['ECG_P_Peaks'][i])
        vector.append(waves_cwt['ECG_P_Offsets'][i])


        vector.append(waves_cwt['ECG_R_Onsets'][i])
        vector.append(waves_cwt['ECG_Q_Peaks'][i])

        vector.append(rpeaks['ECG_R_Peaks'][i])

        vector.append(waves_cwt['ECG_S_Peaks'][i])
        vector.append(waves_cwt['ECG_R_Offsets'][i])


        vector.append(waves_cwt['ECG_T_Onsets'][i])
        vector.append(waves_cwt['ECG_T_Peaks'][i])
        vector.append(waves_cwt['ECG_T_Offsets'][i])

        feature_array.append(vector)
        
    feature_array=np.vstack(feature_array)
    
    feature_array[feature_array==np.nan]
    s=np.isnan(feature_array)
    feature_array[s]=0 
    
    return feature_array
   
def ecg_isoline_drift_correction(ecg_signal, sampling_rate):

    # Apply a high-pass filter to remove baseline wander and DC drift
    b, a = butter(2, 0.5 / (sampling_rate / 2), 'highpass')
    ecg_filtered = filtfilt(b, a, ecg_signal)

    # Estimate the isoelectric line (baseline) using a moving average filter
    window_size = int(sampling_rate * 0.2)  # 200 ms window size
    baseline = savgol_filter(ecg_filtered, window_size, 1)

    # Subtract the estimated baseline from the filtered ECG signal
    ecg_corrected = ecg_filtered - baseline

    return ecg_corrected

def ecg_segmentation(ecg_signal, fs=1000, threshold=0.5):
    # Find R-peaks using a threshold-based approach
    peaks, _ = find_peaks(ecg_signal, height=threshold)
    
    # Calculate the RR intervals
    rr_intervals = np.diff(peaks) / fs


    return peaks, rr_intervals



def extract_ecg_segments(ecg_signal, r_peaks, fs=1000, window_size=0.2):
    # Calculate the window size in samples
    window_size_samples = int(window_size * fs)

    # Initialize an empty array to store the segments
    segments = []

    # Loop over the R-peaks and extract the corresponding segments
    for r_peak in r_peaks:
        start = r_peak - window_size_samples // 2
        end = r_peak + window_size_samples // 2
        segment = ecg_signal[start:end]
        segments.append(segment)
#     return np.array(segments)       
#######################
    lens=[len(s) for s in segments]
    max_len=max(lens)

    new_segements=[]
    for s in segments:
        if len(s)==max_len:
            new_segements.append(s)
        else:
            zeros_size=max_len-len(s)
            s=np.append(s,np.zeros(zeros_size))
            new_segements.append(s)
############################    
    
    
    return np.array(new_segements)



def preprocessing_general(sig):
    ecg_corrected = ecg_isoline_drift_correction(sig, sampling_rate=1000)

    r_peaks, rr_intervals = ecg_segmentation(ecg_corrected, fs=1000, threshold=0.7)
    segments = extract_ecg_segments(ecg_corrected, r_peaks, fs=1000, window_size=0.7)
    return segments


def butter_bandbass_filter(Input_signal,low_cutoff,high_cutoff,sampling_rate,order=4):
    nyq=0.5*sampling_rate #nyquist sampling
    low=low_cutoff/nyq
    high=high_cutoff/nyq
    
    numerator,denominator=butter(order,[low,high],btype='band',output='ba',analog=False,fs=None)
    filtered=filtfilt(numerator,denominator,Input_signal)
    
    return filtered


def preprocessing_11points(sig):
    
    ecg_corrected = ecg_isoline_drift_correction(sig, sampling_rate=1000)
    filtered_signal=butter_bandbass_filter(ecg_corrected,low_cutoff=2,high_cutoff=40,sampling_rate=1000,order=4)
    segments = get_fiducial_features(filtered_signal)
    return segments











