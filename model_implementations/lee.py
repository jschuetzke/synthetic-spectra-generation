
from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model

def cnn_2(input_size=5000, dropout=.5, last_act='softmax',
          dense_neurons=[2000, 500], classes=500, lr=3e-4):
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
    # Hidden Layers
    dense_neurons = dense_neurons if type(dense_neurons) is list else list(dense_neurons)
    for i, neurons in enumerate(dense_neurons):
        if dropout:
            x = layers.Dropout(dropout)(x)
        x = layers.Dense(neurons, activation=None,
                         name=f'dense{i}')(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(classes, activation=last_act, 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(learning_rate=lr)
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
          dense_neurons=[2500, 1000], classes=500, lr=3e-4):
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
    # Hidden Layers
    dense_neurons = dense_neurons if type(dense_neurons) is list else list(dense_neurons)
    for i, neurons in enumerate(dense_neurons):
        if dropout:
            x = layers.Dropout(dropout)(x)
        x = layers.Dense(neurons, activation=None,
                         name=f'dense{i}')(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(classes, activation=last_act, 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(learning_rate=lr)
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