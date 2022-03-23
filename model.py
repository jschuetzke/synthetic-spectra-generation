import numpy as np
from tensorflow.keras import initializers, layers, metrics, optimizers, regularizers
from tensorflow.keras.models import Model

def convnet(input_size=5000, ks=20, conv_filters=4, conv_layers=3, 
            pooling_size=3, pooling_strides=2, layer_do=True, 
            dropout=0.3, classes=50, dense_neurons=[1024, 512],
            hidden_act='relu', reg=0, last_act='sigmoid', opt=None, lr=3e-4):
    #model_name = f'class{classes}_in{input_size}_ks{ks}_c{conv_layers}'
    input_layer = layers.Input(shape=(input_size, 1), name="input")
    x = input_layer
    # Conv Layers
    for i in range(conv_layers):
        x = layers.Conv1D(conv_filters, ks, dilation_rate=2, 
                          padding='same', activation='gelu', 
                          use_bias=False, kernel_initializer='he_uniform', 
                          name=f'conv{i+1}')(x)
        x = layers.MaxPooling1D(pool_size=pooling_size, 
                                strides=pooling_strides,
                                padding='same', 
                                name=f'maxpool{i+1}')(x)
        if layer_do and i < (conv_layers-1):
            x = layers.SpatialDropout1D(0.25, name=f'layer_dropout{i+1}')(x)
    if conv_filters > 1:
        x = layers.Conv1D(1, 1, activation='relu', use_bias=False, name='shrink')(x)
    x = layers.Flatten(name='flat')(x)
    
    # Hidden Layers
    dense_neurons = dense_neurons if type(dense_neurons) is list else list(dense_neurons)
    for i, neurons in enumerate(dense_neurons):
        x = layers.Dense(neurons, activation=None,
                         kernel_regularizer=regularizers.L1(1e-3),
                         name=f'dense{i}')(x)
        if hidden_act == 'leakyrelu':
            x = layers.ReLU(negative_slop=.1)(x)
        else:
            x = layers.Activation(hidden_act)(x)
        if dropout:
            x = layers.Dropout(dropout)(x)
    
    # Output
    out = layers.Dense(classes, activation=last_act, name='output', 
                       use_bias=False, 
                       kernel_regularizer=regularizers.L2(l2=reg))(x)
    model = Model(inputs=input_layer, outputs=out)
    
    # Optimizer and Loss
    if opt is None:
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

def simple(input_size=5000, pooling_blocks=3, pooling_size=3, pooling_strides=2,
           dropout=0, dense_neurons=[256, 256], classes=50, hidden_act='relu',
           normalize_bias=False, reg=0, last_act='softmax', opt=None, lr=3e-4):
    input_layer = layers.Input(shape=(input_size, 1), name="input")
    x = input_layer
    # Pooling blocks
    for i in range(pooling_blocks):
        x = layers.MaxPooling1D(pool_size=pooling_size, strides=pooling_strides,
                                padding='same', 
                                name=f'maxpool{i+1}')(x)
    x = layers.Flatten(name='flat')(x)
    
    # Hidden Layers
    dense_neurons = dense_neurons if type(dense_neurons) is list else list(dense_neurons)
    for i, neurons in enumerate(dense_neurons):
        x = layers.Dense(neurons, activation=None,
                         kernel_regularizer=regularizers.L1(1e-3),
                         name=f'dense{i}')(x)
        if hidden_act == 'leakyrelu':
            x = layers.ReLU(negative_slop=.1)(x)
        else:
            x = layers.Activation(hidden_act)(x)
        if dropout:
            x = layers.Dropout(dropout)(x)
    
    # Output
    bias_init = initializers.Constant(np.log(1./classes)) if normalize_bias else 'zeros'
    out = layers.Dense(classes, activation=last_act, name='output', 
                       bias_initializer=bias_init, 
                       kernel_regularizer=regularizers.L2(l2=reg))(x)
    model = Model(inputs=input_layer, outputs=out)
    
    # Optimizer and Loss
    if opt is None:
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
