import pandas as pd
import numpy as np

def selecte_features(x_train):
    to_clean = []
    to_fit = []
    for features in x_train:
        appened = 0
        for i in x_train[features]:
            if pd.isna(i):
                to_clean.append(features)
                appened = 1
                break
        if appened == 0:
            to_fit.append(features)
    return to_clean, to_fit

def cleaning(x, selected_festures):
    X = pd.DataFrame()
    for feature in selected_festures['to_clean']:
        X[feature] = x[feature].fillna(x[feature].mean())

    for feature in selected_festures['to_fit']:
        X[feature] = x[feature].copy()
    return(X)

def features_scaling(x_train, x_test, selected_features):
    for feature in selected_features:
        _max = pd.concat([x_train[feature], x_test[feature]], ignore_index=True).max()
        _min = pd.concat([x_train[feature], x_test[feature]], ignore_index=True).min()
        x_train[feature] = x_train[feature] / (_max - _min)
        x_test[feature] = x_test[feature] / (_max - _min)
    Bias = np.zeros(x_train.shape[0])
    x_train.insert(loc=0, column='Bias', value=Bias)
    Bias = np.zeros(x_test.shape[0])
    x_test.insert(loc=0, column='Bias', value=Bias)
    return x_train, x_test

