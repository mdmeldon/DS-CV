from sklearn.preprocessing import StandardScaler
import numpy as np

def fit_X_scaler(X_train):
    """fit StandardScaler，and return StandardScaler object
    """
    sc = StandardScaler()
    for _, clips in enumerate(X_train):
        data_i_truncated = np.squeeze(clips)
        sc.partial_fit(data_i_truncated)
    return sc

def get_X_scaled(X_train):
    """apply normlization
    """
    scaler = fit_X_scaler(X_train)
    X_train_new = np.zeros(X_train.shape)
    for indx, clips in enumerate(X_train):
        data_i_truncated = np.squeeze(clips)
        if scaler is not None:  # normlize
            data_i_truncated = scaler.transform(data_i_truncated)
        X_train_new[indx, 0, :, :] = data_i_truncated
    return X_train_new
