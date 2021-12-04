from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import GroupShuffleSplit

from io import BytesIO
from tensorflow.python.lib.io import file_io
import numpy as np
from google.cloud import storage


from sklearn.model_selection import train_test_split

import pandas as pd

from tensorflow.python.keras.layers import Input, Activation,Dense,Dropout, Convolution2D,MaxPool2D,Flatten,BatchNormalization,Convolution1D,MaxPool1D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


import argparse
import os
import numpy as np

import tensorflow as tf
from sklearn import preprocessing


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

def network(learning_rate=0.01):
    im_shape=(12,300,1)
    inputs_cnn=Input(shape=(im_shape), name='inputs_cnn')
    conv1_1=Convolution2D(32, (5,5), activation='relu', padding="same", input_shape=im_shape)(inputs_cnn)
    conv1_1=BatchNormalization()(conv1_1)
    pool1=MaxPool2D(pool_size=(2,2))(conv1_1)
    conv2_1=Convolution2D(64, (5,5), padding="same", activation='relu')(pool1)
    conv2_1=BatchNormalization()(conv2_1)
    pool2=MaxPool2D(pool_size=(2,2))(conv2_1)
    conv3_1=Convolution2D(128, (5,5), padding="same", activation='relu')(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    pool3=MaxPool2D(pool_size=(2,2))(conv3_1)
    flatten=Flatten()(pool3)
    dense_end1 = Dense(128, activation='relu')(flatten)
    dense_end2 = Dense(50, activation='relu')(dense_end1)
    main_output = Dense(1, activation='sigmoid', name='main_output')(dense_end2)
    model = Model(inputs= inputs_cnn, outputs=main_output)
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics = ['accuracy'])

    return model



def network_attiaetal(learning_rate=0.01):
    im_shape=(12,300,1)
    inputs_cnn=Input(shape=(im_shape), name='inputs_cnn')
    convolution_filters=[16,16,32,32,64]
    convolution_sizes=[5,5,5,5,5]
    maxpool_sizes=[1,2,1,2,2]

    x=inputs_cnn
    for i in range(5):
        x = Convolution2D(filters=convolution_filters[i],
                   kernel_size=(1,convolution_sizes[i]),
                   padding='same',
                   activation='linear')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if maxpool_sizes[i]==1:
            continue
        else:
            x = MaxPool2D(pool_size=(1,maxpool_sizes[i]))(x)

    x = Convolution2D(filters=128,
                      kernel_size=(12,1),
                      padding='same',
                      activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(1,2))(x)

    x=Flatten()(x)

    x = Dense(128, activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x=Dropout(0.3)(x)

    x = Dense(64, activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x=Dropout(0.3)(x)


    main_output = Dense(1, activation='sigmoid', name='main_output')(x)
    model = Model(inputs= inputs_cnn, outputs=main_output)
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics = ['accuracy'])

    return model






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

    # all_data_loaded=load_np_array_from_gs_dirs('ecg-data',['100k-data/CSPCData_China/output' ,'100k-data/GeorgiaData_USA/output','100k-data/china_private1/output','100k-data/PTBData_Germany/output'])
    # ,'100k-data/GeorgiaData_USA/output','100k-data/china_private1/output','100k-data/PTBData_Germany/output'
    all_data_loaded=load_np_array_from_gs_dirs('ecg-data',['100k-data/china_private1/output'])

    print('all data size: '+str(all_data_loaded.shape))
    all_data_loaded_df = pd.DataFrame(data=all_data_loaded)

    all_data_loaded_df=all_data_loaded_df[all_data_loaded_df.iloc[:,3658]!=2]
    print('unique values in all: '+str(all_data_loaded_df.iloc[:,3658].count() ))

    # train_df, test_df = train_test_split(all_data_loaded_df, test_size=0.2, random_state=42, shuffle=True)


    #pick the sinus rhythm ones

    all_data_loaded_df=all_data_loaded_df[(all_data_loaded_df.iloc[:,3656]==57.0) | (all_data_loaded_df.iloc[:,3656]==1633.0) ]
    print('all data size: '+str(all_data_loaded_df.shape))
    print('unique patients in sinus only: '+str(all_data_loaded_df.iloc[:,3655].unique().size ))

    train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 42).split(all_data_loaded_df, groups=all_data_loaded_df.iloc[:,3655]))
    train_df = all_data_loaded_df.iloc[train_inds]
    test_df = all_data_loaded_df.iloc[test_inds]
    #
    # train_inds, test_inds = next(StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=52).split(X,y))
    # trainX, testX = X[train_inds], X[test_inds]
    # trainY, testY = y[train_inds], y[test_inds]


    train_df_race_combination=train_df.groupby(train_df.columns[3658]).size().reset_index(name='count')
    print(train_df_race_combination.head())

    test_df_race_combination=test_df.groupby(test_df.columns[3658]).size().reset_index(name='count')
    print(test_df_race_combination.head())

    print('train_df size: '+str(train_df.shape))
    print('test_df size: '+str(test_df.shape))

    train_nparr=train_df.values
    test_nparr=test_df.values

    train_nparr = train_nparr.reshape((train_nparr.shape[0], 12, 305))
    test_nparr = test_nparr.reshape((test_nparr.shape[0], 12, 305))

    trainX=train_nparr[:,:,:300].reshape((-1, 12, 300,1))
    trainY=train_nparr[:,0,303]

    testX=test_nparr[:,:,:300].reshape((-1, 12, 300,1))
    testY=test_nparr[:,0,303]



    print('unique values in testY: '+str(len(np.unique(testY)) ))
    print('unique values in trainY: '+str(len(np.unique(trainY)) ))

    print('X size: '+str(train_nparr.shape))
    print('trainX size: '+str(trainX.shape))
    print('testX size: '+str(testX.shape))


    keras_model = network( learning_rate=args.learning_rate)
    callbacks = [EarlyStopping(monitor='val_loss', patience=8),ModelCheckpoint(filepath='best_model_gender_100k.h5', monitor='val_loss', save_best_only=True)]
    keras_model.fit(trainX, trainY,epochs=50,callbacks=callbacks, batch_size=2000,validation_data=(testX,testY))

    export_path = os.path.join(args.job_dir, 'keras_export')
    tf.keras.experimental.export_saved_model(keras_model, export_path)
    print('Model exported to: {}'.format(export_path))


if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)
