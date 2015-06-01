import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from collections import namedtuple
from time import time

def read_shot_file(fname):
    raw_shot_data = pd.read_csv(fname)
    raw_shot_data.columns = [col.lower() for col in raw_shot_data.columns]
    raw_shot_data.shot_result = raw_shot_data.shot_result == "made"
    return(raw_shot_data)

def print_descriptive_stats(shots_df):
    #descriptive stats
    print("Number of closest defenders: %d"%(shots_df.closest_defender.unique().size))
    print("Number of shooters: %d"%(shots_df.player_name.unique().size))
    print("Fraction of shots made: %f"%(shots_df.shot_result.mean()))
    return

def make_modeling_data(raw_shot_data):
    my_one_hot = OneHotEncoder()
    shooter = raw_shot_data.player_id.values[:, newaxis]
    shooter_as_one_hot = my_one_hot.fit_transform(shooter)

    for_model = pd.DataFrame(data=shooter_as_one_hot.todense(), 
                            columns=my_one_hot.active_features_)
    for_model['shot_dist'] = raw_shot_data.shot_dist
    for_model['close_def_dist'] = raw_shot_data.close_def_dist
    return(for_model, raw_shot_data.shot_result)

def get_model_accuracy(my_model, x, y):
    x=x[:]
    my_model.fit(X=x, y=y)
    return(my_model.score(X=x, y=y))

def define_nn(n_hidden_layers=2, hidden_size=500, dropout_val = 0.25):
    model = Sequential()
    model.add(Dense(283, hidden_size, init='glorot_uniform', activation='relu'))
    model.add(Dropout(dropout_val))
    for i in range(n_hidden_layers):
        model.add(BatchNormalization((hidden_size,)))
        model.add(PReLU((hidden_size,)))
        if dropout_val>0:
            model.add(Dropout(dropout_val))
    model.add(Dense(hidden_size, 1, init='glorot_uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")
    return model


if __name__ == "__main__":
    raw_fname = "../data/shot_logs.csv"
    raw_shot_data = read_shot_file(raw_fname)
    print_descriptive_stats(raw_shot_data)
    x, y = make_modeling_data(raw_shot_data)
    nn = define_nn(4, 500, 0)
    nn_hist = nn.fit(x.values, y.values, nb_epoch=10, batch_size=1300, 
            validation_split=0.3, show_accuracy=True)
    model = LogisticRegression()
    print("Logistic accuracy: %f"%(get_model_accuracy(model, x, y)))