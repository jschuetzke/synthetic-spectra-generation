
from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model

def vgg(input_size=5000, dropout=0, last_act='softmax',
        dense_neurons=[120, 84, 186], classes=500, lr=3e-4):
    input_layer = layers.Input(shape=(input_size, 1),
                               name="input")
    x = layers.Conv1D(6, 5, strides=1, padding='same',
                      activation='relu', 
                      name='conv1')(input_layer)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool1')(x)
    x = layers.Dropout(dropout, name='dropout1')(x)
    x = layers.Conv1D(16, 5, strides=1, padding='same',
                      activation='relu', 
                      name='conv2_1')(x)
    x = layers.Conv1D(16, 5, strides=1, padding='same',
                      activation='relu', 
                      name='conv2_2')(x)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool2')(x)
    x = layers.Dropout(dropout, name='dropout2')(x)
    x = layers.Conv1D(32, 5, strides=1, padding='same',
                      activation='relu', 
                      name='conv3_1')(x)
    x = layers.Conv1D(32, 5, strides=1, padding='same',
                      activation='relu', 
                      name='conv3_2')(x)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool3')(x)
    x = layers.Dropout(dropout, name='dropout3')(x)
    x = layers.Conv1D(64, 5, strides=1, padding='same',
                      activation='relu', 
                      name='conv4_1')(x)
    x = layers.Conv1D(64, 5, strides=1, padding='same',
                      activation='relu', 
                      name='conv4_2')(x)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool4')(x)
    x = layers.Dropout(dropout, name='dropout4')(x)
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
