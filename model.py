import math 
from math import sqrt
import numpy as np 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from numpy import mean, absolute
import time

class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cumulative time taken
        self.timetaken = time.process_time()
    def on_epoch_end(self,epoch,logs = {}):
        self.times.append((epoch, time.process_time() - self.timetaken))
    def on_train_end(self,logs = {}):
        previous_time = 0
        for item in self.times:
            print("Epoch ", item[0], " run time is: ", item[1]-previous_time)
            previous_time = item[1]
        print("Total trained time is: ", previous_time)


def LSTM_encoder_decoder(train,n_output_steps,n_input_steps,neuron,lr,batch,epochs):
    X_train = []
    Y_train = []

    # Loop for training data
    # Index: 0 (Open), 1 (High), 2 (Low), ...3 (Close)
    for i in range(n_input_steps,train.shape[0] - n_output_steps + 1):
        X_train.append(train[i-n_input_steps:i]) # List
        Y_train.append(train[i + n_output_steps - 1:i + n_output_steps, 3]) # <== Index 3 (CLOSE)
    X_train, Y_train = np.array(X_train), np.array(Y_train) # Transform list to numpy array

    timetaken = timecallback()
    # Setting up an early stop
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=80,  verbose=1, mode='min')
    callbacks_list = [earlystop, timetaken]

    # Use Keras sequential model
    encoder_decoder = tf.keras.Sequential()
    
    # Use Keras sequential model
    encoder_decoder = tf.keras.Sequential()
    
    # Encoder LSTM layer with Dropout regularisation; Set return_sequences to False since we are feeding last output to decoder layer
    encoder_decoder.add(tf.keras.layers.LSTM(neuron, input_shape = (X_train.shape[1],X_train.shape[2]), return_sequences=False))
    encoder_decoder.add(tf.keras.layers.Dropout(0.2))
    
    # The fixed-length output of the encoder is repeated, once for each required time period in the output sequence with the RepeatVector wrapper
    encoder_decoder.add(tf.keras.layers.RepeatVector(n_output_steps))

    encoder_decoder.add(tf.keras.layers.LSTM(neuron, return_sequences=True))
    encoder_decoder.add(tf.keras.layers.Dropout(0.2))
    
    # Decoder LSTM layer with Dropout regularisation; Set return_sequences to False to feed each output time period to a Dense layer
    encoder_decoder.add(tf.keras.layers.LSTM(neuron, return_sequences=False))
    encoder_decoder.add(tf.keras.layers.Dropout(0.2))

    # Output layer
    encoder_decoder.add(tf.keras.layers.Dense(1, activation='linear'))

    # Compile the model
    encoder_decoder.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr), loss = 'mean_squared_error')
    print(encoder_decoder.summary())

    # Training the data
    history_en_dec = encoder_decoder.fit(X_train,Y_train,epochs=epochs,batch_size=batch,validation_split=0.1,verbose = 1,
                                        shuffle = False, callbacks=callbacks_list)
    encoder_decoder.reset_states()
    return encoder_decoder, history_en_dec

# Formula for caculate MAPE with zero values
def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mean_absolute_percentage_error(data, y_pred): 
    return np.round(np.mean(np.abs(percentage_error(np.asarray(data), np.asarray(y_pred)))) * 100, 4)

# Evaluating the model
def evaluate_model(model,test,n_output_steps,n_input_steps):
    X_test = []
    Y_test = []

    # Loop for testing data
    # Index: 0 (Open), 1 (High), 2 (Low), ...3 (Close)
    for i in range(n_input_steps,test.shape[0] - n_output_steps +1):
        X_test.append(test[i-n_input_steps:i]) # List
        Y_test.append(test[i + n_output_steps - 1:i + n_output_steps, 3]) # <== Index 3 (CLOSE)
    X_test,Y_test = np.array(X_test), np.array(Y_test) # Transform list to numpy array

    # Prediction Time !!!!
    Yhat = model.predict(X_test)
    mse_acc = np.round(mean_squared_error(Y_test,Yhat), 4)
    mad_acc = np.round(mean(absolute(Y_test - mean(Yhat))), 4)
    mape_acc = mean_absolute_percentage_error(Y_test, Yhat)
    return  mse_acc, mad_acc, mape_acc, X_test, Y_test, Yhat