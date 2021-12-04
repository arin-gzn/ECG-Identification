from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from io import BytesIO
from tensorflow.python.lib.io import file_io
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.python.keras.layers import Input,UpSampling2D ,Activation,Dense, Convolution2D,MaxPool2D,Flatten,BatchNormalization,Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


import argparse
import os
import numpy as np

import tensorflow as tf
from sklearn import preprocessing
from google.cloud import storage



def network(learning_rate=0.01):
    im_shape=(12,300,1)
    inputs_cnn=Input(shape=(im_shape), name='inputs_cnn')

    x=Convolution2D(128, (5,5), activation='relu', padding="same", input_shape=im_shape)(inputs_cnn)
    x=MaxPool2D(pool_size=(2,2), padding='same')(x)
    x=Convolution2D(64, (5,5), padding="same", activation='relu')(x)
    x=MaxPool2D(pool_size=(2,2), padding='same')(x)
    x=Convolution2D(32, (5,5), padding="same", activation='relu')(x)
    encoded=MaxPool2D(pool_size=(2,2), padding='same')(x)

    x = Convolution2D(32, (5, 5), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(64, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(128, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded= Convolution2D(1, (5, 5), activation='sigmoid')(x)

    model = Model(inputs= inputs_cnn, outputs=decoded)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])

    return model


def network2_linear_output(learning_rate=0.01):
    im_shape=(12,300,1)
    inputs_cnn=Input(shape=(im_shape), name='inputs_cnn')

    x=Convolution2D(128, (5,5), activation='relu', padding="same", input_shape=im_shape)(inputs_cnn)
    x=MaxPool2D(pool_size=(2,2), padding='same')(x)
    x=Convolution2D(64, (5,5), padding="same", activation='relu')(x)
    x=MaxPool2D(pool_size=(2,2), padding='same')(x)
    x=Convolution2D(32, (5,5), padding="same", activation='relu')(x)
    encoded=MaxPool2D(pool_size=(2,2), padding='same')(x)

    x = Convolution2D(32, (5, 5), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(64, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(128, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded= Convolution2D(1, (5, 5), activation='linear')(x)

    model = Model(inputs= inputs_cnn, outputs=decoded)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])

    return model







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


def load_and_scale(bucket,dir_list,scaler_type='MaxMin'):
    all_data_loaded=load_np_array_from_gs_dirs(bucket,dir_list)
    print('all data size cspchina: '+str(all_data_loaded.shape))
    all_data_nparr=all_data_loaded[~np.any(np.isnan(all_data_loaded) , axis=1)]
    all_data_nparr=all_data_nparr.reshape((all_data_nparr.shape[0], 12, 305))
    X=all_data_nparr[:,:,:300]
    X=X.reshape(X.shape[0],12*300)
    if scaler_type=='Standard':
        scaler = StandardScaler()
        scaler.fit(X)
        X=scaler.transform(X)
    elif scaler_type=='MaxMin':
        x_min = X.min()
        x_max = X.max()
        print('max min seen in data cspchina: '+str((x_min,x_max)))
        X = (X - x_min)/(x_max-x_min)
    elif scaler_type=='None':
        pass

    else:
        print('Error: Wrong scaler type!')

    y=all_data_nparr[:,0,300]
    X=X.reshape((-1, 12, 300, 1))

    return X,y

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





def train_and_evaluate(args):

    X,y=load_and_scale('ecg-data',['100k-data/china_private1/output' ])

    train_inds, test_inds = next(StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=52).split(X,y))
    trainX, testX = X[train_inds], X[test_inds]

    keras_model = network2_linear_output( learning_rate=args.learning_rate)
    callbacks = [EarlyStopping(monitor='val_loss', patience=8),ModelCheckpoint(filepath='best_model_100k_identification.h5', monitor='val_loss', save_best_only=True)]
    keras_model.fit(trainX, trainX,epochs=200,callbacks=callbacks, batch_size=1000,validation_data=(testX,testX))

    export_path = os.path.join(args.job_dir, 'keras_export')
    tf.keras.experimental.export_saved_model(keras_model, export_path)
    print('Model exported to: {}'.format(export_path))


if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)
