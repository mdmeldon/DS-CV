# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#import torchvision
# from models.resnet import resnet50
from auc import AUCCallback
import torchvision
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


if __name__ == '__main__':

    BATCH_SIZE = 16
    N_CLASS = 12

    
    DIR_DATA = Path(__file__).absolute().parent.parent / 'data'
    ref = pd.read_csv("data/processed/Data_path.csv")
    mfcc = np.load('data/processed/MFCC_PREPARE_AUG.npy')

    # Split between train and test 
    X_train, X_test, y_train, y_test = train_test_split(mfcc,
                                                        ref.labels,
                                                        test_size=0.25,
                                                        shuffle=True,
                                                        random_state=8
                                                    )

    # one hot encode the target 
    lb = LabelEncoder()

    y_train = lb.fit_transform(y_train) # to_categorical
    y_test = lb.fit_transform(y_test) # to_categorical

    # Normalization as per the standard NN process
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train = (X_train - mean)/std
    X_test = (X_test - mean)/std

    classes_names = list(lb.classes_)

    model = torchvision.models.resnet50(pretrained=True,progress=True)

    # Изменение количества выходных классов 
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
                                torch.nn.Dropout(0.2),
                                torch.nn.Linear(num_ftrs, N_CLASS)
    )
    model.maxpool = torch.nn.Sequential(
                                        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                        torch.nn.Dropout(0.2)

    # Изменение количества входных слоёв
    model.conv1 = torch.nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
    featuresTrain = torch.FloatTensor(X_train[0:-1])
    targetsTrain = torch.from_numpy(y_train[0:-1]).type(torch.LongTensor) # data type is long

    # create feature and targets tensor for test set.
    featuresTest = torch.FloatTensor(X_test[0:-1])
    targetsTest = torch.from_numpy(y_test[0:-1]).type(torch.LongTensor) # data type is long

    trainSet = TensorDataset(featuresTrain,targetsTrain)
    validSet = TensorDataset(featuresTest,targetsTest)

    # data loader
    train_loader = DataLoader(trainSet, batch_size = BATCH_SIZE, shuffle = True)
    valid_loader = DataLoader(validSet, batch_size = BATCH_SIZE, shuffle = False)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Init catalyst components
    runner = SupervisedRunner(device=device)
    loaders = {'train': train_loader, 'valid': valid_loader}
    logdir = 'data/logs/0'
    callbacks = [
                AUCCallback(num_classes = 12, class_names= classes_names),
                AccuracyCallback(),
                ConfusionMatrixCallback(num_classes = 12, class_names= classes_names),
    ]

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        callbacks=callbacks,
        logdir=logdir,
        num_epochs=120,
        verbose=True,        
    )  