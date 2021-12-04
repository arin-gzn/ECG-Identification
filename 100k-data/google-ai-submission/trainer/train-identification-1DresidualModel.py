from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from io import BytesIO
from tensorflow.python.lib.io import file_io
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from tensorflow.python.keras.layers import Input, Dense, Convolution2D,MaxPool2D,Flatten,BatchNormalization
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


import argparse
import os
import numpy as np

import tensorflow as tf
from sklearn import preprocessing
from google.cloud import storage

from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dropout, AveragePooling1D,
                                     BatchNormalization, Activation, Add, ZeroPadding1D, Conv2D, MaxPool2D, AveragePooling2D, MaxPooling2D, GlobalAveragePooling1D,
                                     Flatten, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform, constant
import tensorflow as tf
import numpy as np



def load_np_array_from_gs_dirs(bucket_name,gs_dir_list):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    files_processed=0
    for gs_dir in gs_dir_list:
        for blob in bucket.list_blobs(prefix=gs_dir):
            if blob.name.endswith(".csv"):
                files_processed=files_processed+1
                file_full_path='gs://'+bucket_name+'/'+blob.name
                print('processing: ', blob.name)
                f = BytesIO(file_io.read_file_to_string(file_full_path, binary_mode=True))
                np_datat_loaded = np.loadtxt(f, delimiter=',')
                if files_processed==1:
                    combined_data = np_datat_loaded
                else:
                    combined_data=np.concatenate((combined_data, np_datat_loaded), axis=0)

    return combined_data



def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting '
             'models')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=.01,
        type=float,
        help='learning rate for gradient descent, default=.01')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args



def convolutional_block(X, CH, filtersize=15):
    """
    Implementation of the convolutional block

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv1D(filters=CH, kernel_size=filtersize, strides=2, padding = 'same',  kernel_initializer="he_normal")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv1D(filters=CH, kernel_size=filtersize, strides =1, padding = 'same', kernel_initializer="he_normal")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv1D(filters=CH, kernel_size=filtersize, strides =1, padding = 'same', kernel_initializer="he_normal")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv1D(filters=CH, kernel_size=filtersize, strides=2, padding = 'same', kernel_initializer="he_normal")(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def identity_block(X, CH, filtersize=15):
    """
    Implementation of the identity block

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv1D(filters=CH, kernel_size=filtersize, strides=1, padding='same', kernel_initializer="he_normal")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv1D(filters=CH, kernel_size=filtersize, strides=1, padding='same', kernel_initializer="he_normal")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv1D(filters=CH, kernel_size=filtersize, strides=1, padding='same', kernel_initializer="he_normal")(X)
    X = BatchNormalization()(X)

    ##### SHORTCUT PATH ####
    #X_shortcut = Conv1D(filters=CH, kernel_size=filtersize, strides=2, kernel_initializer="he_normal")(X_shortcut)
    #X_shortcut = BatchNormalization()(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def OneDResRawDataModle(input_shape = (5000,3),  output_bias=None):
    X_input = Input(input_shape)
    #X = ZeroPadding1D(padding=60)(X_input)
    X = Conv1D(filters=64, kernel_size=15, strides=2, name='conv1', kernel_initializer="he_normal")(X_input)
    X = BatchNormalization(name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(pool_size=2, strides=2)(X)

    X = convolutional_block(X, 64, filtersize=15)
    #X = MaxPooling1D(pool_size=2, strides=2)(X)
    X = identity_block(X, 64, filtersize=15)
    X = identity_block(X, 64, filtersize=15)
    X = identity_block(X, 64, filtersize=15)
    #X = convolutional_block(X, 64, filtersize=15)
    #X = MaxPooling1D(pool_size=2, strides=2)(X)
    #X = identity_block(X, 64, filtersize=15)
    #X = convolutional_block(X, 64, filtersize=15)
    #X = MaxPooling1D(pool_size=2, strides=2)(X)
    #X = identity_block(X, 64, filtersize=15)

    X = convolutional_block(X, 128, filtersize=7)
    #X = MaxPooling1D(pool_size=2, strides=2)(X)
    X = identity_block(X, 128, filtersize=7)
    #X = convolutional_block(X, 128, filtersize=7)
    X = identity_block(X, 128, filtersize=7)
    #X = convolutional_block(X, 128, filtersize=7)
    X = identity_block(X, 128, filtersize=7)
    #X = identity_block(X, 128, filtersize=7)
    #X = identity_block(X, 128, filtersize=7)

    X = convolutional_block(X, 256, filtersize=5)
    #X = MaxPooling1D(pool_size=2, strides=2)(X)
    X = identity_block(X, 256, filtersize=5)
    X = identity_block(X, 256, filtersize=5)
    X = identity_block(X, 256, filtersize=5)
    #X = convolutional_block(X, 256, filtersize=5)
    #X = identity_block(X, 256, filtersize=5)
    #X = convolutional_block(X, 256, filtersize=5)
    #X = identity_block(X, 256, filtersize=5)


    X = convolutional_block(X, 512, filtersize=3)
    #X = MaxPooling1D(pool_size=2, strides=2)(X)
    X = identity_block(X, 512, filtersize=3)

    X = identity_block(X, 512, filtersize=3)
    X = identity_block(X, 512, filtersize=3)

    X = AveragePooling1D(2,2)(X)
    X = Flatten()(X)
    X = Dense(512, activation="relu")(X)
    X = Dropout(0.001)(X)
    X = Dense(512, activation="relu")(X)
    #X = Dense(256, activation="relu")(X)
    #X = Dense(128, activation="relu")(X)
    X = Dense(48, activation='sigmoid')(X)
    model = Model(inputs=X_input, outputs = X)
    return model

def OneDResRawDataModle001(input_shape = (5000,3),  output_bias=None):
    X_input = Input(input_shape)
    #X = ZeroPadding1D(padding=60)(X_input)
    X = Conv1D(filters=64, kernel_size=15, strides=2, name='conv1', kernel_initializer="he_normal")(X_input)
    X = BatchNormalization(name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(pool_size=2, strides=2)(X)

    X = convolutional_block(X, 64, filtersize=15)
    #X = MaxPooling1D(pool_size=2, strides=2)(X)
    X = identity_block(X, 64, filtersize=15)
    #X = identity_block(X, 64, filtersize=15)
    X = identity_block(X, 64, filtersize=15)
    X = convolutional_block(X, 64, filtersize=15)
    #X = MaxPooling1D(pool_size=2, strides=2)(X)
    #X = identity_block(X, 64, filtersize=15)
    #X = convolutional_block(X, 64, filtersize=15)
    #X = MaxPooling1D(pool_size=2, strides=2)(X)
    #X = identity_block(X, 64, filtersize=15)

    X = convolutional_block(X, 128, filtersize=7)
    #X = MaxPooling1D(pool_size=2, strides=2)(X)
    X = identity_block(X, 128, filtersize=7)
    #X = convolutional_block(X, 128, filtersize=7)
    #X = identity_block(X, 128, filtersize=7)
    #X = convolutional_block(X, 128, filtersize=7)
    #X = identity_block(X, 128, filtersize=7)
    #X = identity_block(X, 128, filtersize=7)
    #X = identity_block(X, 128, filtersize=7)

    X = convolutional_block(X, 256, filtersize=5)
    #X = MaxPooling1D(pool_size=2, strides=2)(X)
    X = identity_block(X, 256, filtersize=5)
    #X = identity_block(X, 256, filtersize=5)
    #X = identity_block(X, 256, filtersize=5)
    #X = convolutional_block(X, 256, filtersize=5)
    #X = identity_block(X, 256, filtersize=5)
    #X = convolutional_block(X, 256, filtersize=5)
    #X = identity_block(X, 256, filtersize=5)


    X = convolutional_block(X, 512, filtersize=3)
    X = MaxPooling1D(pool_size=2, strides=2)(X)
    X = identity_block(X, 512, filtersize=3)

    #X = identity_block(X, 512, filtersize=3)
    #X = identity_block(X, 512, filtersize=3)

    X = AveragePooling1D(2,2)(X)
    X = Flatten()(X)
    X = Dense(512, activation="relu")(X)
    X = Dropout(0.0001)(X)
    X = Dense(512, activation="relu")(X)
    #X = Dense(256, activation="relu")(X)
    #X = Dense(128, activation="relu")(X)
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
        X = Dense(1, activation='sigmoid', bias_initializer=output_bias)(X)
    else:
        X = Dense(13, activation='sigmoid')(X)
    model = Model(inputs=X_input, outputs = X)
    return model


def OneDResRawDataModle002(input_shape = (5000,8), nb_classes=13):

    n_feature_maps = 64
    input_layer = Input(input_shape)

    # BLOCK 1

    conv_x = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_1 = Add()([shortcut_y, conv_z])
    output_block_1 = Activation('relu')(output_block_1)
    # BLOCK 2

    conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_2 = Add()([shortcut_y, conv_z])
    output_block_2 = Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = BatchNormalization()(output_block_2)

    output_block_3 = Add()([shortcut_y, conv_z])
    output_block_3 = Activation('relu')(output_block_3)

    # FINAL
    gap_layer = GlobalAveragePooling1D()(output_block_3)
    output_layer = Dense(nb_classes, activation='sigmoid')(gap_layer)
    #13, activation='sigmoid'
    model = Model(inputs=input_layer, outputs=output_layer)
    return model
    #model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
    #                metrics=['accuracy'])

    #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

    #file_path = self.output_directory + 'best_model.hdf5'

    #model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
    #                                                    save_best_only=True)

    #self.callbacks = [reduce_lr, model_checkpoint]






def train_and_evaluate(args):

    # all_data_loaded=load_np_array_from_gs_dirs('ecg-data',['100k-data/CSPCData_China/output' ])
    # print('all data size cspchina: '+str(all_data_loaded.shape))
    # all_data_nparr=all_data_loaded[~np.any(np.isnan(all_data_loaded) , axis=1)]
    # all_data_nparr = all_data_nparr.reshape((all_data_nparr.shape[0], 12, 305))
    # X=all_data_nparr[:,:,:300].reshape((-1, 12, 300, 1))
    # y_cspchina=all_data_nparr[:,0,300]
    # x_min = X.min()
    # x_max = X.max()
    # X_scaled_cspchina = (X - x_min)/(x_max-x_min)
    # print('max min seen in data cspchina: '+str((x_min,x_max)))

    # all_data_loaded=load_np_array_from_gs_dirs('ecg-data',['100k-data/GeorgiaData_USA/output' ])
    # print('all data size usa: '+str(all_data_loaded.shape))
    # all_data_nparr=all_data_loaded[~np.any(np.isnan(all_data_loaded) , axis=1)]
    # all_data_nparr = all_data_nparr.reshape((all_data_nparr.shape[0], 12, 305))
    # X=all_data_nparr[:,:,:300].reshape((-1, 12, 300, 1))
    # x_min = X.min()
    # x_max = X.max()
    # X_scaled_georgiausa = (X - x_min)/(x_max-x_min)
    # y_georgiausa=all_data_nparr[:,0,300]
    # print('max min seen in data usa: '+str((x_min,x_max)))

    all_data_loaded=load_np_array_from_gs_dirs('ecg-data',['100k-data/china_private1/output' ])
    print('all data size china private1: '+str(all_data_loaded.shape))
    all_data_nparr=all_data_loaded[~np.any(np.isnan(all_data_loaded) , axis=1)]
    all_data_nparr = all_data_nparr.reshape((all_data_nparr.shape[0], 12, 305))
    X=all_data_nparr[:,:,:300].reshape((-1, 12, 300))
    y_china_private1=all_data_nparr[:,0,300]
    x_min = X.min()
    x_max = X.max()
    X_scaled_china_private1 = (X - x_min)/(x_max-x_min)
    print('max min seen in data china private1: '+str((x_min,x_max)))

    # all_data_loaded=load_np_array_from_gs_dirs('ecg-data',['100k-data/PTBData_Germany/output' ])
    # print('all data size germany: '+str(all_data_loaded.shape))
    # all_data_nparr=all_data_loaded[~np.any(np.isnan(all_data_loaded) , axis=1)]
    # all_data_nparr = all_data_nparr.reshape((all_data_nparr.shape[0], 12, 305))
    # X=all_data_nparr[:,:,:300].reshape((-1, 12, 300, 1))
    # y_germany=all_data_nparr[:,0,300]
    # x_min = X.min()
    # x_max = X.max()
    # X_scaled_germany = (X - x_min)/(x_max-x_min)
    # print('max min seen in data germany: '+str((x_min,x_max)))

    #
    # X=np.concatenate(( X_scaled_georgiausa,X_scaled_china_private1), axis=0)
    # X_scaled_cspchina=None
    # X_scaled_georgiausa=None
    # X_scaled_china_private1=None
    # X_scaled_germany=None
    #
    # y=np.concatenate(( y_georgiausa,y_china_private1), axis=0)
    # y_cspchina=None
    # y_georgiausa=None
    # y_china_private1=None

    X=X_scaled_china_private1
    y=y_china_private1


    print('X  sample row: '+str(X[10]))
    print('X  last element: '+str(X[299]))
    print('y  sample row: '+str(y[10]))


    train_inds, test_inds = next(StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42).split(X,y))
    trainX, testX = X[train_inds], X[test_inds]
    trainY, testY = y[train_inds], y[test_inds]


    print('unique values in all: '+str(len(np.unique(y)) ))
    print('unique values in testY: '+str(len(np.unique(testY)) ))
    print('unique values in trainY: '+str(len(np.unique(trainY)) ))

    print('X size: '+str(X.shape))
    print('trainX size: '+str(trainX.shape))
    print('testX size: '+str(testX.shape))

    le = preprocessing.LabelEncoder()
    le.fit(np.concatenate((trainY,testY)))
    train_y_encoded = le.transform(trainY)
    test_y_encoded = le.transform(testY)

    num_of_people_in_data=np.unique(testY).shape[0]
    print('test_df size: '+str(num_of_people_in_data))
    trainX=trainX.transpose(0,2,1)
    testX=testX.transpose(0,2,1)
    # Create the Keras Model
    keras_model = OneDResRawDataModle002( input_shape=(300,12),nb_classes=num_of_people_in_data)
    keras_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics = ['sparse_categorical_accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=8),ModelCheckpoint(filepath='best_model_100k_identification.h5', monitor='val_loss', save_best_only=True)]
    keras_model.fit(trainX, train_y_encoded,epochs=200,callbacks=callbacks, batch_size=2000,validation_data=(testX,test_y_encoded))
    # model.load_weights('best_model_100k_identification.h5')

    export_path = os.path.join(args.job_dir, 'keras_export')
    tf.keras.experimental.export_saved_model(keras_model, export_path)
    print('Model exported to: {}'.format(export_path))


if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)
