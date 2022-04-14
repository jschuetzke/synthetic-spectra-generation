
from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model
from .basic_blocks import conv1d

def cnn(input_size=5000, dropout=.5, last_act='softmax',
        dense_neurons=[2048], classes=500, lr=3e-4):
    input_layer = layers.Input(shape=(input_size, 1),
                               name="input")
    # Batch Norm between conv and activation so we use custom conv1d func
    # instead of default keras Conv1D layer
    x = conv1d(input_layer, 16, 21, activation='leakyrelu', batch_norm=True)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool1')(x)
    x = conv1d(x, 32, 11, activation='leakyrelu', batch_norm=True)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool2')(x)
    x = conv1d(input_layer, 64, 5, activation='leakyrelu', batch_norm=True)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool3')(x)
    x = layers.Flatten(name='flat')(x)
    # Hidden Layers
    dense_neurons = dense_neurons if type(dense_neurons) is list else list(dense_neurons)
    for i, neurons in enumerate(dense_neurons):
        if dropout:
            x = layers.Dropout(dropout)(x)
        x = layers.Dense(neurons, activation='tanh',
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