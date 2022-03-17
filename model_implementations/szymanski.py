
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model

def cnn(input_size=5000, dropout=.7, last_act='softmax',
	    classes=500, lr=3e-4):
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
    x = layers.Dropout(dropout, name='dropout1')(x)
    x = layers.Dense(3100, activation='relu',
                     name='dense1')(x)
    x = layers.Dropout(dropout, name='dropout2')(x)
    x = layers.Dense(1200, activation='relu',
                     name='dense2')(x) # None activation
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
