# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#import torchvision
# from models.resnet import resnet50
from auc import AUCCallback

import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst.dl import SupervisedRunner
from catalyst.dl.utils import set_global_seed, prepare_cudnn
from catalyst.dl.callbacks import AccuracyCallback, PrecisionRecallF1ScoreCallback, VerboseLogger, ConfusionMatrixCallback

# Other  
from tqdm import tqdm, tqdm_pandas
import scipy
from scipy.stats import skew
import librosa.display
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import pandas as pd
import seaborn as sns
import glob 
import os
import sys
import IPython.display as ipd  # To play sound in the notebook
import warnings
from pathlib import Path
# ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from ...dirs import DIR_DATA_MODELS, DIR_DATA_INTERHIM

if __name__ == '__main__':

    import torch
    import torchvision

    # model = AudioRNN()

    # checkpoint = torch.load(str(DIR_DATA_MODELS / 'SoundMFCCParams.pth'))
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()

    path_file = str(DIR_DATA_INTERHIM / 'AIMAF/audio/0.wav')
    inputs = create_dataset_mfcc(path_file)
    print(torch.max(model(inputs), 1)[1].data)
    print(torch.max(model(inputs), 1)[0].data)