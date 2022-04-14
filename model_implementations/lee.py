
from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model
from .basic_blocks import get_dense_stack

def cnn_2(input_size=5000, dropout=.5, dense_neurons=[2000, 500],
          classes=500, lr=3e-4):
    input_layer = layers.Input(shape=(input_size, 1),
                               name="input")
    x = layers.Conv1D(64, 50, strides=2, padding='same',
                      activation='relu', 
                      name='conv1')(input_layer)
    x = layers.MaxPool1D(3, strides=2, 
                         name='maxpool1')(x)
    x = layers.Conv1D(64, 25, strides=3, padding='same',
                      activation='relu', 
                      name='conv2')(x)
    x = layers.MaxPool1D(2, strides=3, 
                         name='maxpool2')(x)
    x = layers.Flatten(name='flat')(x)
    x = get_dense_stack(x, dense_neurons, dropout)
    out = layers.Dense(classes, activation='softmax', 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=[metrics.CategoricalAccuracy(name='accuracy')])
    return model

def cnn_3(input_size=5000, dropout=.3, dense_neurons=[2500, 1000], 
          classes=500, lr=3e-4):
    input_layer = layers.Input(shape=(input_size, 1), 
                               name="input")
    x = layers.Conv1D(64, 20, strides=1, padding='same',
                      activation='relu', 
                      name='conv1')(input_layer)
    x = layers.MaxPool1D(3, strides=3, 
                         name='maxpool1')(x)
    x = layers.Conv1D(64, 15, strides=1, padding='same',
                      activation='relu', 
                      name='conv2')(x)
    x = layers.MaxPool1D(2, strides=3, 
                         name='maxpool2')(x)
    x = layers.Conv1D(64, 10, strides=2, padding='same',
                      activation='relu', 
                      name='conv3')(x)
    x = layers.MaxPool1D(1, strides=2, 
                         name='maxpool3')(x)
    x = layers.Flatten(name='flat')(x)
    x = get_dense_stack(x, dense_neurons, dropout)
    out = layers.Dense(classes, activation='softmax', 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=[metrics.CategoricalAccuracy(name='accuracy')])
    return model