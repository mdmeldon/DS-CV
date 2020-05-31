# Import libraries 
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import torch
import pandas as pd
import glob 
from sklearn.metrics import confusion_matrix
import IPython.display as ipd  # To play sound in the notebook
import os
import sys
import warnings
from pathlib import Path
# ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def SAVEE_markup(dir_path):
    # Get the data location for SAVEE
    dir_list = os.listdir(dir_path)

    emo_dict = {'_a': 'male_angry', '_d': 'male_disgust', '_f': 'male_fear',
                '_h': 'male_happy', '_n': 'male_neutral', 'sa': 'male_sad'}
    # parse the filename to get the emotions
    emotion=[]
    path = []
    for i in dir_list:
        if i[-8:-6] in emo_dict:
            emotion.append(emo_dict[i[-8:-6]])
        # else:
        #     emotion.append('male_error') 
        path.append(dir_path + i)
    
    # Now check out the label count distribution 
    SAVEE_df = pd.DataFrame(emotion, columns = ['labels'])
    SAVEE_df['source'] = 'SAVEE'
    SAVEE_df = pd.concat([SAVEE_df, pd.DataFrame(path, columns = ['path'])], axis = 1)
    return SAVEE_df

def RAV_markup(dir_path):
    dir_list = os.listdir(dir_path)
    dir_list.sort()

    emotion = []
    gender = []
    path = []
    for i in dir_list:
        fname = os.listdir(dir_path + i)
        for f in fname:
            part = f.split('.')[0].split('-')
            if (int(part[2]) != 8):
                emotion.append(int(part[2]))
                temp = int(part[6])
                if temp%2 == 0:
                    temp = "female"
                else:
                    temp = "male"
                gender.append(temp)
                path.append(dir_path + i + '/' + f)
            
    RAV_df = pd.DataFrame(emotion)
    RAV_df = RAV_df.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust'})
    RAV_df = pd.concat([pd.DataFrame(gender),RAV_df],axis=1)
    RAV_df.columns = ['gender','emotion']
    RAV_df['labels'] =RAV_df.gender + '_' + RAV_df.emotion
    RAV_df['source'] = 'RAVDESS'  
    RAV_df = pd.concat([RAV_df,pd.DataFrame(path, columns = ['path'])],axis=1)
    RAV_df = RAV_df.drop(['gender', 'emotion'], axis=1)
    return RAV_df

def TESS_markup(dir_path):
    #The speakers and the emotions are organised in seperate folders which is very convenient
    dir_list = os.listdir(dir_path)
    dir_list.sort()
    path = []
    emotion = []

    emo_dict = {'an': 'female_angry', 'di': 'female_disgust', 'fe': 'female_fear', 'ha': 'female_happy',
                'ne': 'female_neutral', 'sa': 'female_sad'}
    for i in dir_list:
        fname = os.listdir(dir_path + i)
        print(i.lower())
        for f in fname:
            now_emotional = i.lower()[4:6] 
            if  now_emotional in emo_dict:
                emotion.append(emo_dict[now_emotional])
            # else:
            #     emotion.append('Unknown')
            path.append(dir_path + i + "/" + f)

    TESS_df = pd.DataFrame(emotion, columns = ['labels'])
    TESS_df['source'] = 'TESS'
    TESS_df = pd.concat([TESS_df,pd.DataFrame(path, columns = ['path'])],axis=1)
    return TESS_df

def CREMA_markup(dir_path):
    dir_list = os.listdir(dir_path)
    dir_list.sort()

    gender = []
    emotion = []
    path = []
    female = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,
            1052,1053,1054,1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,1082,1084,1089,1091]

    emo_dict = {'SAD': '_sad', 'ANG': '_angry', 'DIS': '_disgust', 'FEA': '_fear',
                'HAP': '_happy', 'NEU': '_neutral'}

    for i in dir_list: 
        part = i.split('_')
        if int(part[0]) in female:
            temp = 'female'
        else:
            temp = 'male'
        gender.append(temp)
        if part[2] in emo_dict:
            emotion.append(temp + emo_dict[part[2]])
        else:
            emotion.append('Unknown')
        path.append(dir_path + i)
        
    CREMA_df = pd.DataFrame(emotion, columns = ['labels'])
    CREMA_df['source'] = 'CREMA'
    CREMA_df = pd.concat([CREMA_df,pd.DataFrame(path, columns = ['path'])],axis=1)
    return CREMA_df

if __name__ == '__main__':
    # PATHS FILES
    DIR_DATA = Path(__file__).absolute().parent.parent / 'data'
    DATA_RAW = DIR_DATA / 'raw'
    TESS = f'{DATA_RAW}/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/'
    RAV = f'{DATA_RAW}/ravdess-emotional-speech-audio/audio_speech_actors_01-24/'
    SAVEE = f'{DATA_RAW}/surrey-audiovisual-expressed-emotion-savee/ALL/'
    CREMA = f'{DATA_RAW}/cremad/AudioWAV/'

    SAVEE_df = SAVEE_markup(SAVEE)
    RAV_df = RAV_markup(RAV)
    TESS_df = TESS_markup(TESS)
    CREMA_df = CREMA_markup(CREMA)

    df = pd.concat([SAVEE_df, RAV_df, TESS_df, CREMA_df], axis = 0)
    df.to_csv(DIR_DATA / 'processed/Data_path.csv',index=False)
    SAVEE_df.to_csv(DIR_DATA / 'interim/SAVEE_df.csv',index=False)
    RAV_df.to_csv(DIR_DATA / 'interim/RAV_df.csv',index=False)
    TESS_df.to_csv(DIR_DATA / 'interim/TESS_df.csv',index=False)
    CREMA_df.to_csv(DIR_DATA / 'interim/CREMA_df.csv',index=False)
