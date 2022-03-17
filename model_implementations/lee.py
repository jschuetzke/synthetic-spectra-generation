
from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model

def cnn_2(input_size=5000, dropout=.5, last_act='softmax',
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
    x = layers.Dropout(dropout, name='dropout1')(x)
    x = layers.Dense(2000, name='dense1')(x) # None activation
    x = layers.Dropout(dropout, name='dropout2')(x)
    x = layers.Dense(500, name='dense2')(x) # None activation
    x = layers.Dropout(dropout, name='dropout3')(x)
    out = layers.Dense(classes, activation=last_act, 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(lr=lr)
    if last_act == 'softmax':
        loss_fn = 'categorical_crossentropy'
        metrics_list = [metrics.CategoricalAccuracy(name='accuracy')]
    else:
        loss_fn = 'binary_crossentropy'
        metrics_list = [metrics.BinaryAccuracy(name='accuracy'), 
                        metrics.Recall(), metrics.Precision()]
    model.compile(optimizer=opt, loss=loss_fn,
                  metrics=metrics_list)
    return model

def cnn_3(input_size=5000, dropout=.3, last_act='softmax',
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
    x = layers.Dropout(dropout, name='dropout1')(x)
    x = layers.Dense(2500, name='dense1')(x) # None activation
    x = layers.Dropout(dropout, name='dropout2')(x)
    x = layers.Dense(1000, name='dense2')(x) # None activation
    x = layers.Dropout(dropout, name='dropout3')(x)
    out = layers.Dense(classes, activation=last_act, 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(lr=lr)
    if last_act == 'softmax':
        loss_fn = 'categorical_crossentropy'
        metrics_list = [metrics.CategoricalAccuracy(name='accuracy')]
    else:
        loss_fn = 'binary_crossentropy'
        metrics_list = [metrics.BinaryAccuracy(name='accuracy'), 
                        metrics.Recall(), metrics.Precision()]
    model.compile(optimizer=opt, loss=loss_fn,
                  metrics=metrics_list)
    return model