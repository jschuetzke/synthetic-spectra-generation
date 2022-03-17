
from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model

def vgg(input_size=5000, dropout=.7, last_act='softmax',
        classes=500, lr=3e-4):
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
    out = layers.Flatten(name='flat')(x)
    out = layers.Dense(120, activation='relu',
                       kernel_initializer='he_uniform',
                       name='dense1')(out)
    out = layers.Dense(84, activation='relu',
                       kernel_initializer='he_uniform',
                       name='dense2')(out)
    out = layers.Dense(186, activation='relu',
                       kernel_initializer='he_uniform',
                       name='dense3')(out)
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
