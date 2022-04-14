#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import model_implementations as models
import wandb
from wandb.keras import WandbCallback

tf.config.experimental.enable_op_determinism()

if tf.config.list_physical_devices('GPU'):
    # enable memory growth instead of blocking whole VRAM
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

classes = np.unique(ytest).size
yt = tf.one_hot(yt, classes)
yv = tf.one_hot(yv, classes)
ytest = tf.one_hot(ytest, classes)

class TestMetrics(tf.keras.callbacks.Callback):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        super().__init__()
        
    def on_epoch_end(self, epoch, logs=None):
        logs = {} or logs
        res = self.model.evaluate(self.x, self.y, verbose=0)
        for metric, result in zip(self.model.metrics_names,res):
            m_name = 'test_'+metric
            logs[m_name] = result
        super().on_epoch_end(epoch, logs)

batch_size=128

model_list = ['cnn2', 'cnn3', 'cnn6', 'vgg', 'resnet', 'cnn_bn', 'inc3', 'inc6']

for model_name in model_list:
    for seed in range(5):
        wandb.init(project="synthetic-benchmark")
        wandb.config.update({'model_type' : model_name, 'seed' : seed,
                             'batch_size' : batch_size})
        callbacks = [EarlyStopping(patience=25, verbose=1,
                                   restore_best_weights=True, min_delta=0.0001),
                     ReduceLROnPlateau(patience=10, verbose=1),
                     TestMetrics(xtest, ytest),
                     WandbCallback(save_model=False, save_graph=False)]
        tf.keras.utils.set_random_seed(seed)
        model = getattr(models, model_name)()
        # make sure order of data is identical during training
        tf.keras.utils.set_random_seed(seed)
        model.fit(xt, yt, batch_size=batch_size, epochs=500, verbose=2, 
                  callbacks=callbacks, validation_data=(xv, yv), shuffle=True)
