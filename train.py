#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model import convnet, simple
from sklearn.preprocessing import LabelBinarizer

seed = 1

tf.keras.utils.set_random_seed(seed)
tf.config.experimental.enable_op_determinism()

# enable memory growth instead of blocking whole VRAM
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.set_visible_devices(physical_devices[0], 'GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

# import data
xt = np.load('x_train.npy')
xt /= np.max(xt, axis=1, keepdims=True)
xv = np.load('x_val.npy')
xv /= np.max(xv, axis=1, keepdims=True)
xtest = np.load('x_test.npy')
xtest /= np.max(xtest, axis=1, keepdims=True)

yt = np.load('y_train.npy')
yv = np.load('y_val.npy')
ytest = np.load('y_test.npy')

lb = LabelBinarizer()
lb.fit(yt)

yt = lb.transform(yt)
yv = lb.transform(yv)
ytest = lb.transform(ytest)

batch_size=16

callbacks = [EarlyStopping(patience=25, verbose=1,
                           restore_best_weights=True, min_delta=0.0001),
             ReduceLROnPlateau(patience=10, verbose=1)]

model = simple(pooling_blocks=7, dense_neurons=[256, 256], reg=1e-5, last_act='sigmoid')
#model = convnet(last_act='softmax')
model.fit(xt, yt, epochs=500, verbose=2, callbacks=callbacks, validation_data=(xv, yv), shuffle=True)
tf.keras.models.save_model(model, 'test_simple1.h5')
