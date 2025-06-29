import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch 
from tqdm import tqdm
import os
import librosa
from ruamel.yaml import YAML
import scipy.signal as ss
import sys

def replace_indent(stream):
    stream = "     " + stream
    return stream.replace("\n", "\n     ")

yaml = YAML()
yaml.indent(mapping = 2, sequence=2, offset = 2)
yaml.default_flow_style = False
with open('./configs/test.yaml', 'r') as f:
    cfg = yaml.load(f)

with open('./configs/test.yaml', 'w') as f:
    cfg['dataset']['rootdir'] = './results'
    yaml.dump(cfg, f)
    yaml.dump(cfg, sys.stdout, transform=replace_indent)


