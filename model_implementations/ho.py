
from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model
from .basic_blocks import residual_block
        
def resnet(input_size=5000, filters=100, layer_num=6, blocks_per_layer=2, 
           batch_norm=True, last_act='softmax', dense_neurons=[], 
           dropout=.5, classes=500, lr=3e-4):
    input_layer = layers.Input(shape=(input_size, 1),
                               name="input")
    # first regular conv
    x = layers.Conv1D(64, 5, strides=1, padding='same', 
                      name='conv1')(input_layer) # No activation (linear)
    x = layers.BatchNormalization()(x)
    for l in range(layer_num):
        for b in range(blocks_per_layer):
            block_type = 'conv' if b == 0 else 'identity'
            x = residual_block(x, block_type=block_type, batch_norm=batch_norm)
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
            


