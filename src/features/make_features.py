import librosa 
import numpy as np

'''
1. Data Augmentation method   
'''
def speedNpitch(data):
    """
    Speed and Pitch Tuning.
    """
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high = 1)
    speed_fac = 1.2  / length_change # try changing 1.0 to 2.0 ... =D
    tmp = np.interp(np.arange(0,len(data),speed_fac),np.arange(0,len(data)),data)
    minlen = min(data.shape[0], tmp.shape[0])
    data *= 0
    data[0:minlen] = tmp[0:minlen]
    return data

'''
2. Extracting the MFCC feature as an image (Matrix format).  
'''
def make_feature_extraction(file_path, n_params= 30, aug= 1, sampling_rate= 44100, audio_duration= 2.5, mfcc_true= 1):
    
    X_params = np.empty(shape=(1, n_params, 216))
    input_length = sampling_rate * audio_duration
    data, _ = librosa.load(file_path, 
                            sr=sampling_rate,
                            res_type="kaiser_fast",
                            duration=2.5, # Продолжительность
                            offset=0.5 # Начать считывание с 0.5
                            )

    # Random offset / Padding
    if len(data) > input_length:
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length+offset)]
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")

    # Augmentation? 
    if aug == 1:
        data = speedNpitch(data)
    
    # which feature?

    if mfcc_true == 1:
        # MFCC extraction 
        MFCC = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_params)
        MFCC = np.expand_dims(MFCC, axis=0)
        X_params = MFCC
        
    else:
        # Log-melspectogram
        melspec = librosa.feature.melspectrogram(data, n_mels = n_params)   
        logspec = librosa.amplitude_to_db(melspec)
        logspec = np.expand_dims(logspec, axis=0)
        X_params = logspec

    return X_params
        


