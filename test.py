import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch 
from tqdm import tqdm
import os
import librosa
from ruamel.yaml import YAML
import scipy.signal as ss

x = [[file for file in os.listdir('./dataset')]]
x[0].append([file for file in os.listdir('./results')])
print(x)

# with h5py.File('./dataset/Glasgow/1 December 2017 Dataset/1P56A01R02_1.h5', 'r') as hf:
#     r_heatmap = hf['r_heatmap'][()]
#     rv_heatmap = hf['rv_heatmap'][()]
#     label = hf['label'][()]
#     person = hf['person'][()]
#     radar_param = hf['radar_param'][()]

# rv_heatmap = torch.abs(torch.from_numpy(rv_heatmap))
# print((rv_heatmap[0,1]))

# print(radar_param)
# print(label,person)
# fs = radar_param[4]
# rmax = radar_param[5]
# vmax = radar_param[6]
# plt.subplot(211)
# dB_plot = librosa.power_to_db(np.abs(rv_heatmap[:,:])**2)
# librosa.display.specshow(np.asanyarray(dB_plot), sr=fs, x_axis='time', y_axis='linear', cmap='jet')
# plt.subplot(212)
# dB_plot = librosa.power_to_db(np.abs(r_heatmap[:,:])**2)
# librosa.display.specshow(np.asanyarray(dB_plot), sr=fs, x_axis='time', y_axis='linear', cmap='jet')

# plt.show()

# yaml = YAML()
# yaml.indent(mapping = 2, sequence=2, offset = 2)
# yaml.default_flow_style = False
# with open('./configs/config.yaml', 'r') as f:
#     cfg = yaml.load(f)

# ninput = cfg['model']['num_inputs'] 
# print([int(item.strip()) for item in ninput.split(',')])

# def replace_indent(stream):
#     stream = "     " + stream
#     return stream.replace("\n", "\n     ")












