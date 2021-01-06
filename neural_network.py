#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import layers, models, regularizers, backend, utils
import keras.backend.tensorflow_backend as tfback
from IPython.display import display, clear_output

#importing sydney dataset(remember to replace first parameter with acutal file name or path)
sydney_df = pd.read_csv(csvfile, 
                        names = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16',
                                 'Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Y10','Y11','Y12','Y13','Y14','Y15','Y16',
                                 'P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P_total'])
Xs = rows[:,0:32]
ys = rows[:,32:49]
val_ratio = 0.2
N = Xs.shape[0]
Xs_train = Xs[0:int((1-val_ratio)*N)]
ys_train = ys[0:int((1-val_ratio)*N)]
Xs_val = Xs[int((1-val_ratio)*N):]
ys_val = ys[int((1-val_ratio)*N):]
print(Xs_train.shape,ys_train.shape)
print(Xs_val.shape,ys_val.shape)

#neural network hyperparameter value
hyperparamter_penalty_weight = 1e2

#y_predict is the predicted power output from the neural network 
def custom_loss(y_true,y_pred):
  return K.mean(K.square(y_pred - y_true), axis=-1) + hyperparamter_penalty_weight * K.mean(K.square(K.sum(y_pred[:,0:16], axis=1) - y_pred[:,16]), axis=-1)

def penalty(y_true,y_pred):
  return 100*K.mean(K.abs(K.sum(y_pred[:,0:16], axis=1) - y_pred[:,16])/K.abs(y_pred[:,16]), axis=-1)

inputs = layers.Input(shape=(32,))
d1 = layers.Dense(128, activation = 'relu')(inputs)
d2 = layers.Dense(128, activation = 'relu')(d1)
d3 = layers.Dense(128, activation = 'relu')(d2)
outputs = layers.Dense(17, activation = 'linear')(d3)

simple_model = models.Model(inputs = inputs, outputs = outputs)
simple_model.compile('adam', custom_loss, metrics = ['mean_absolute_percentage_error'])
simple_model.summary()

class CustomCallback(keras.callbacks.Callback):
    def on_test_end(self, logs = None):
        clear_output()

loss_hist = simple_model.fit(Xs_train, ys_train, epochs = 500, validation_data = (Xs_val, ys_val), batch_size=500)

train_err = loss_hist.history['mean_absolute_percentage_error']
val_err = loss_hist.history['val_mean_absolute_percentage_error']
plt.plot(np.arange(0,len(train_err)),train_err,'-')
plt.plot(np.arange(0,len(val_err)),val_err,'-')
plt.show()

pred = simple_model.predict(Xs_val)
penalty(ys_val, pred).numpy()
