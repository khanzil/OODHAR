import os
import shutil
import numpy as np
from numpy.fft import fft, fftshift
import h5py
import matplotlib.pyplot as plt
import librosa
import scipy
import scipy.signal as ss

def range_velocity_extractor(data: np.ndarray, rmax, vmax, n_fft_range, n_fft_dop, win_size_dop = 200, window_dop = 'hann', 
                             overlap_dop = 0.95, static_remove = None, filter = None):
    """
        Use this to extract feature maps of FMCW.

        Input:

            data : np.ndarray, complex matrix size (fast_time, slow_time)
            n_fft : int, length of the spectogram, size 2
            static_remove : str, using some algorithm to remove the signal caused by static objects

        Output:

            velocity_range_fft: complex matrix size ()
            

    """
    if n_fft_range == None:
        n_fft_range = data.shape[0]
    if n_fft_dop == None:
        n_fft_dop = win_size_dop
    range_fft = fftshift(fft(data, n=n_fft_range, axis=0), axes=0)[-n_fft_range//2+1:,:]

    if filter == 'butter':
        sos = ss.butter(4,0.0075,'high',output='sos')
        range_fft = ss.sosfilt(sos, range_fft, axis = 1)

    if static_remove == 'avg':
        range_fft = range_fft - np.average(range_fft, axis=1)

    _,_,velocity_range_fft = ss.stft(range_fft, nperseg=win_size_dop, window=window_dop, axis=1, 
                                          noverlap=win_size_dop*overlap_dop, return_onesided=False)
    
    velocity_range_fft = fftshift(velocity_range_fft, axes=1).T

    return velocity_range_fft, range_fft

fold_list = [fold for fold in os.listdir("./dataset/Glasgow") if os.path.isdir('./dataset/Glasgow/'+fold)]
write_to_h5 = []
for fold in fold_list:
    file_list = [file for file in os.listdir("./dataset/Glasgow/"+fold) if ".dat" in file]
    for file in file_list:
        label = file[file.find('A')+1:file.find('A')+3]
        person = file[file.find('P')+1:file.find('P')+3]
        repetition = file[file.find('R')+1:file.find('R')+3]
        print(label,person,repetition)
        with open('./dataset/Glasgow/'+fold+'/'+file,'r') as datfile:
            raw_str = datfile.read()
        raw_str = raw_str.splitlines()

        fc = float(raw_str[0])                      # center freq
        Tsweep = float(raw_str[1])/1000             # length of each chirp/sweep time in s
        fast_time = int(raw_str[2])                 # number of samples per chirp/number of time samples per sweep
        Bw = float(raw_str[3])                      # bandwidth
        data = np.array([complex(x.replace('i','j')) for x in raw_str[4:]])
        slow_time = int(len(data)/fast_time)        # num_chirps
        fs = fast_time/Tsweep
        data = data.reshape(slow_time, fast_time).T
        # plt.plot(np.real(data[0,:]))

        print(fc, Tsweep, fast_time, Bw, slow_time, fs)

        rv_heatmap, r_heatmap = range_velocity_extractor(data,rmax=0,vmax=0,n_fft_range=None,n_fft_dop=None,filter='butter')
        # plt.plot(20*np.log10(np.abs(r_heatmap[:,200])))
        plt.subplot(121)
        dB_plot = librosa.power_to_db(np.abs(rv_heatmap)**2)
        librosa.display.specshow(np.asanyarray(dB_plot), sr=fs, x_axis='time', y_axis='linear', cmap='jet')
        plt.subplot(122)
        dB_plot = librosa.power_to_db(np.abs(r_heatmap)**2)
        librosa.display.specshow(np.asanyarray(dB_plot), sr=fs, x_axis='time', y_axis='linear', cmap='jet')
        plt.show()
        raise ValueError
        



        























