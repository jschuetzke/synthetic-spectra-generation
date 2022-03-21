
from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model

def cnn(input_size=5000, dropout=.7, last_act='softmax',
        dense_neurons=[3100, 1200], classes=500, lr=3e-4):
    input_layer = layers.Input(shape=(input_size, 1), 
                               name="input")
    x = layers.Conv1D(64, 35, strides=1, padding='same',
                      activation='relu', 
                      name='conv1')(input_layer)
    x = layers.MaxPool1D(3, strides=2, 
                         name='maxpool1')(x)
    x = layers.Conv1D(64, 30, strides=1, padding='same',
                      activation='relu', 
                      name='conv2')(x)
    x = layers.MaxPool1D(3, strides=2, 
                         name='maxpool2')(x)
    x = layers.Conv1D(64, 25, strides=1, padding='same',
                      activation='relu', 
                      name='conv3')(x)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool3')(x)
    x = layers.Conv1D(64, 20, strides=1, padding='same',
                      activation='relu', 
                      name='conv4')(x)
    x = layers.MaxPool1D(1, strides=2, 
                         name='maxpool4')(x)
    x = layers.Conv1D(64, 15, strides=1, padding='same',
                      activation='relu', 
                      name='conv5')(x)
    x = layers.MaxPool1D(1, strides=2, 
                         name='maxpool5')(x)
    x = layers.Conv1D(64, 10, strides=1, padding='same',
                      activation='relu', 
                      name='conv6')(x)
    x = layers.MaxPool1D(1, strides=2, 
                         name='maxpool6')(x)
    x = layers.Flatten(name='flat')(x)
    # Hidden Layers
    dense_neurons = dense_neurons if type(dense_neurons) is list else list(dense_neurons)
    for i, neurons in enumerate(dense_neurons):
        if dropout:
            x = layers.Dropout(dropout)(x)
        x = layers.Dense(neurons, activation='relu',
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
