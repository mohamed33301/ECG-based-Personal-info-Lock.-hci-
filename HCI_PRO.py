#!/usr/bin/env python
# coding: utf-8

# 1. check data 
# 2. check fiducial features

# In[696]:


# segmentation 👌done
# ui tkinter test saved in a separated file ,not preprcessed before 👌done 
# fiducial features 👌done 


# In[697]:


# reveiw project and test again  👌done
# save some array in text files to read later for testing ok 👌done
# then tkinter


# In[373]:


import os
import numpy as np
import pandas as pd
import scipy.io
import scipy

from statsmodels.graphics import tsaplots
import statsmodels.api as sm
from scipy.signal import butter,filtfilt,savgol_filter

import matplotlib.pyplot as plt
# from scipy import signal
from scipy.signal import butter,filtfilt
import warnings
warnings.filterwarnings('ignore')
import glob
import wfdb as wf


# In[374]:


s2=r'D:\last semaster\Human Computer Interactions\Authentication-system-based-on-ECG-main\subjects\p_156\s0299lre'
s3=r'D:\last semaster\Human Computer Interactions\Authentication-system-based-on-ECG-main\subjects\p_165\s0322lre'
s5=r'D:\last semaster\Human Computer Interactions\Authentication-system-based-on-ECG-main\subjects\p_174\s0300lre'
s7=r'D:\last semaster\Human Computer Interactions\Authentication-system-based-on-ECG-main\subjects\p_184\s0363lre'
s8=r'D:\last semaster\Human Computer Interactions\Authentication-system-based-on-ECG-main\subjects\p_185\s0336lre'
s9=r'D:\last semaster\Human Computer Interactions\Authentication-system-based-on-ECG-main\subjects\p_198\s0402lre'
s10=r'D:\last semaster\Human Computer Interactions\Authentication-system-based-on-ECG-main\subjects\sub_150\s0287lre'
s13=r'D:\last semaster\Human Computer Interactions\Authentication-system-based-on-ECG-main\subjects\sub260\s0496_re'


# In[375]:


import numpy as np
import scipy.signal as signal

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


# In[376]:


def butter_bandbass_filter(Input_signal,low_cutoff,high_cutoff,sampling_rate,order=4):
    nyq=0.5*sampling_rate #nyquist sampling
    low=low_cutoff/nyq
    high=high_cutoff/nyq
    
    numerator,denominator=butter(order,[low,high],btype='band',output='ba',analog=False,fs=None)
    filtered=filtfilt(numerator,denominator,Input_signal)
    
    return filtered


# In[377]:


from scipy.signal import find_peaks
def ecg_segmentation(ecg_signal, fs=1000, threshold=0.5):
    # Find R-peaks using a threshold-based approach
    peaks, _ = find_peaks(ecg_signal, height=threshold)

    
    # Calculate the RR intervals
   # rr_intervals = np.diff(peaks) / fs



    return peaks


# In[415]:


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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # primary for the whole signal 

# # start project code

# In[416]:


files=[s2,s5,s13,s9,]#s8
# [s2,s3,s5,s7,s9,s10,s8,s13]#are the best after long analysis
files


# In[417]:


data={}
for idx,sub in enumerate(files):
    signal_array, fields=wf.rdsamp(sub)
    sig = signal_array[:,1]
    sname=f"sub_{idx+1}"
    data[sname]=sig


# In[418]:


data


# # split data to get newtest unseen data
# train =[:80000]
# 
# test =[80000:85000]
# 

# In[419]:


# save testing
new_test={}
for key ,sig in data.items():
    new_test[key]=sig[80000:90000]
    np.savetxt(f'{key}_test_segment.txt',sig[80000:90000])


# In[420]:


def preprocessing_general(sig):
    ecg_corrected = ecg_isoline_drift_correction(sig, sampling_rate=1000)

    r_peaks = ecg_segmentation(ecg_corrected, fs=1000, threshold=0.7)
    segments = extract_ecg_segments(ecg_corrected, r_peaks, fs=1000, window_size=0.7)
    return segments


# In[421]:


segments_dict={}
for key ,sig in data.items():
    print(key,sig)
    segments = preprocessing_general(sig)
    segments_dict[key] = segments[2:52]


# In[422]:


# fs = 1000
# cutoff_low = 1
# cutoff_high = 40
# nyquist = fs / 2
# level = int(np.floor(np.log2(nyquist/cutoff_low)))
# level


# In[ ]:





# - We have extracted fiducial features related to QRS complex using Tompkin’s algorithm. The reason to select features related to only QRS complex is that QRS complex is considered to be fairly constant and doesn’t change with the change of heart rate as it reflects the depolarization of ventricular muscle
# 
# https://web.archive.org/web/20160911023941/http://ijarcsse.com/docs/papers/Volume_5/7_July2015/V5I6-0385.pdf

# In[423]:


from fiducial_features import pan_tompkins
def preprocess_using_fiducial(filtered_signal):
    
    features=pan_tompkins(filtered_signal)
    return features


# In[424]:


# according to the paper
    # db8,level 5
    # Daubechies sym7
# https://accentsjournals.org/PaperDirectory/Journal/IJACR/2020/3/3.pdf


# In[ ]:





# In[425]:


def get_features(signal):
    


    filtered_signal=butter_bandbass_filter(signal,low_cutoff=2,high_cutoff=40,sampling_rate=1000,order=4)
    

    
   
  
    features=preprocess_using_fiducial(filtered_signal)
  
    return features


# In[426]:


def get_sub(idx):
    return f'subject_{idx+1}'


# In[427]:


labels=[]
final_data=[]
type_=2
for key ,sig in segments_dict.items():
    for s in sig:
        s=get_features(s)
        labels.append(key)
        final_data.append(s)

final_data=np.vstack(final_data)


final_data


# # split data into train and test

# In[428]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X_train, X_test, y_train, y_test = train_test_split(final_data,labels, test_size=0.20,stratify=labels,shuffle=True,random_state=42)

lb=LabelEncoder()

y_train=lb.fit_transform(y_train)
y_test=lb.transform(y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# # SVM

# In[429]:


from sklearn import svm
from sklearn.metrics import accuracy_score ,classification_report,ConfusionMatrixDisplay,confusion_matrix


# In[430]:


svm_classifier = svm.SVC(kernel='linear') # Linear Kernel

svm_classifier.fit(X=X_train,y=y_train)

pred=svm_classifier.predict(X_test)

print("Accuracy = {} % ".format(accuracy_score(y_test,pred)*100))


# In[431]:


print(classification_report(y_test,pred,target_names=pd.unique(labels)))


# In[432]:


cm=confusion_matrix(y_test,pred)
disp=ConfusionMatrixDisplay(cm)
disp.plot()


# # LogisticRegression

# In[433]:


from sklearn.linear_model import LogisticRegression

LR_classifier=LogisticRegression(max_iter=1000)

LR_classifier.fit(X=X_train,y=y_train)

pred=LR_classifier.predict(X_test)

print("Accuracy = {} % ".format(accuracy_score(y_test,pred)*100))


# In[434]:


cm=confusion_matrix(y_test,pred)
disp=ConfusionMatrixDisplay(cm)
disp.plot()


# # LDA

# In[435]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[436]:


LDA_classifier=LinearDiscriminantAnalysis()

LDA_classifier.fit(X=X_train,y=y_train)

pred=LDA_classifier.predict(X_test)

print("Accuracy = {} % ".format(accuracy_score(y_test,pred)*100))


# In[437]:


cm=confusion_matrix(y_test,pred)
disp=ConfusionMatrixDisplay(cm)
disp.plot()


# In[ ]:





# In[ ]:





# In[ ]:





# # saving models

# In[ ]:





# In[369]:


# type_=1:use wavelet
# type_=2:use fiducial_features
# type_=3:use AC/DCT


# In[330]:


import pickle


# In[331]:


pickle.dump(LDA_classifier,open('LDA_classifier_fiducial.pkl','wb'))
    


# # Loading and Testing

# In[332]:


path=r'C:\Users\hp\Downloads\project_hci\sub_4_test_segment.txt' 
to_test = np.loadtxt(path)


# In[333]:


to_test.shape


# In[334]:


loaded_model=pickle.load(open('LDA_classifier_fiducial.pkl','rb'))


# In[335]:


loaded_model.n_features_in_


# In[336]:


to_test.shape


# In[337]:


to_test=preprocessing_general(to_test)


# In[338]:


to_test.shape


# In[339]:


to_test=to_test[2]


# In[340]:


test=get_features(to_test)


# In[341]:


test=np.array(test)
test.shape


# In[342]:


def prep_type(type_):
  
 
    return 'fiducial_features'
  


# In[343]:


plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.title('signal')
plt.plot(to_test)

plt.subplot(2,1,2)
# plt.title(f'After preprocessing using{prep_type(type_)}')
# plt.plot(test[0])

if type_==2:
    plt.title(f'After preprocessing using{prep_type(type_)}')
    plt.plot(test[0])

else:
    
    plt.title(f'After preprocessing using{prep_type(type_)}')
    plt.plot(test)


# In[344]:


if type_ !=2:
    test=np.expand_dims(test,axis=0)
print('test shape after ',test.shape)
test.shape


# In[345]:


pred=loaded_model.predict(test)

pred[0]


# In[346]:


get_sub(pred[0])


# In[ ]:





# In[ ]:





# In[251]:





# In[ ]:




