from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from io import BytesIO
from tensorflow.python.lib.io import file_io
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from tensorflow.python.keras.layers import Input, Dense, Convolution2D,MaxPool2D,Flatten,BatchNormalization,Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from datetime import datetime
import json

import argparse
import os
import numpy as np

import tensorflow as tf
from sklearn import preprocessing
from google.cloud import storage



def network_spatiotempo(num_of_people_in_data,learning_rate=0.01):
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
                          activation='relu')(x)
        x = BatchNormalization()(x)
        if maxpool_sizes[i]==1:
            continue
        else:
            x = MaxPool2D(pool_size=(1,maxpool_sizes[i]))(x)

    x = Convolution2D(filters=128,
                      kernel_size=(12,1),
                      padding='same',
                      activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(1,2))(x)

    x=Flatten()(x)

    x = Dense(128, activation='relu')(x)
    # x=Dropout(0.1)(x)

    x = Dense(64, activation='relu')(x)
    # x=Dropout(0.1)(x)

    main_output = Dense(num_of_people_in_data, activation='softmax', name='main_output')(x)
    model = Model(inputs= inputs_cnn, outputs=main_output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics = ['sparse_categorical_accuracy'])

    return model


def network(num_of_people_in_data,learning_rate=0.01):
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
    main_output = Dense(num_of_people_in_data, activation='softmax', name='main_output')(dense_end2)
    model = Model(inputs= inputs_cnn, outputs=main_output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics = ['sparse_categorical_accuracy'])

    return model


def network2(num_of_people_in_data,learning_rate=0.01):
    im_shape=(12,300,1)
    inputs_cnn=Input(shape=(im_shape), name='inputs_cnn')
    conv1_1=Convolution2D(32, (5,5), activation='relu', padding="same", input_shape=im_shape)(inputs_cnn)
    conv1_1=BatchNormalization()(conv1_1)
    pool1=MaxPool2D(pool_size=(2,2))(conv1_1)
    conv2_2=Convolution2D(64, (5,5), padding="same", activation='relu')(pool1)
    conv2_2=BatchNormalization()(conv2_2)
    pool2=MaxPool2D(pool_size=(2,2))(conv2_2)
    dropout1=Dropout(0.3)(pool2)
    conv4_1=Convolution2D(128, (5,5), padding="same", activation='relu')(dropout1)
    conv4_1=BatchNormalization()(conv4_1)
    pool4=MaxPool2D(pool_size=(2,2))(conv4_1)
    dropout3=Dropout(0.5)(pool4)
    flatten=Flatten()(dropout3)
    dense_end1 = Dense(512, activation='relu')(flatten)
    dropout1=Dropout(0.5)(dense_end1)
    dense_end1 = Dense(256, activation='relu')(dropout1)
    dense_end2 = Dense(100, activation='relu')(dense_end1)
    main_output = Dense(num_of_people_in_data, activation='softmax', name='main_output')(dense_end2)
    model = Model(inputs= inputs_cnn, outputs=main_output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics = ['sparse_categorical_accuracy'])
    print(model.summary())
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


def load_and_scale(bucket,dir_list,dataset_id,scaler_type='None'):
    all_data_loaded=load_np_array_from_gs_dirs(bucket,dir_list)
    print('all data size cspchina: '+str(all_data_loaded.shape))
    all_data_nparr=all_data_loaded[~np.any(np.isnan(all_data_loaded) , axis=1)]
    all_data_nparr=all_data_nparr.reshape((all_data_nparr.shape[0], 12, 305))
    X=all_data_nparr[:,:,:300]
    y=all_data_nparr[:,0,300]

    if scaler_type!='None':
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
        return

    X=X.reshape((-1, 12, 300, 1))
    heartbeat_attributes_all=all_data_nparr[:,0,300:305]
    dataset_id_array=np.repeat(dataset_id,heartbeat_attributes_all.shape[0]).reshape((heartbeat_attributes_all.shape[0],1))
    heartbeat_attributes_all=np.concatenate([heartbeat_attributes_all,dataset_id_array ], axis=1)

    return X,y, heartbeat_attributes_all


def add_str(x,my_prefix):
    return "{}-{}".format(x, my_prefix)


def train_and_evaluate(args):

    X_scaled_cspchina,y_cspchina,attributes_all_cspchina=load_and_scale('ecg-data',['100k-data/CSPCData_China/output' ],0)
    X_scaled_georgiausa,y_georgiausa,attributes_all_georgiausa=load_and_scale('ecg-data',['100k-data/GeorgiaData_USA/output' ],1)
    X_scaled_china_private1,y_china_private1,attributes_all_china_private1=load_and_scale('ecg-data',['100k-data/china_private1/all/output' ],2)
    X_scaled_germany,y_germany,attributes_all_germany=load_and_scale('ecg-data',['100k-data/PTBData_Germany/output' ],3)

    X=np.concatenate(( X_scaled_georgiausa,X_scaled_china_private1,X_scaled_cspchina,X_scaled_germany), axis=0)
    X_scaled_cspchina=None
    X_scaled_georgiausa=None
    X_scaled_china_private1=None
    X_scaled_germany=None


    print('min patientid in y_china_private1: ' +str(y_china_private1.min()))
    print('max patientid in y_china_private1: ' +str(y_china_private1.max()))

    print('min patientid in y_cspchina: ' +str(y_cspchina.min()))
    print('max patientid in y_cspchina: ' +str(y_cspchina.max()))

    print('min patientid in y_georgiausa: ' +str(y_georgiausa.min()))
    print('max patientid in y_georgiausa: ' +str(y_georgiausa.max()))

    print('min patientid in y_germany: ' +str(y_germany.min()))
    print('max patientid in y_germany: ' +str(y_germany.max()))

    vfunc = np.vectorize(add_str)
    y_georgiausa=vfunc(y_georgiausa.astype(str),'usa')
    print('y_georgiausa  y: '+str(y_georgiausa[0:10] ))
    y_china_private1=vfunc(y_china_private1.astype(str),'chinapriv')
    print('y_china_private1  y: '+str(y_china_private1[0:10] ))
    y_cspchina=vfunc(y_cspchina.astype(str),'cspchina')
    print('y_cspchina  y: '+str(y_cspchina[0:10] ))
    y_germany=vfunc(y_germany.astype(str),'germany')
    print('y_germany  y: '+str(y_germany[0:10] ))


    y=np.concatenate(( y_georgiausa,y_china_private1,y_cspchina,y_germany), axis=0)

    print('unique values in y_georgiausa: '+str(len(np.unique(y_georgiausa)) ))
    print('unique values in y_china_private1: '+str(len(np.unique(y_china_private1)) ))
    print('unique values in y_cspchina: '+str(len(np.unique(y_cspchina)) ))
    print('unique values in y_germany: '+str(len(np.unique(y_germany)) ))
    print('unique values in y: '+str(len(np.unique(y)) ))
    y_cspchina=None
    y_georgiausa=None
    y_china_private1=None
    y_germany=None


    attributes_all=np.concatenate(( attributes_all_georgiausa,attributes_all_china_private1,attributes_all_cspchina,attributes_all_germany), axis=0)
    attributes_all_georgiausa=None
    attributes_all_china_private1=None
    attributes_all_cspchina=None
    attributes_all_germany=None



    train_inds, test_inds = next(StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42).split(X,y))
    trainX, testX = X[train_inds], X[test_inds]
    trainY, testY = y[train_inds], y[test_inds]
    attributes_all_test=attributes_all[test_inds]


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

    print('min patientid in test_y_encoded: ' +str(test_y_encoded.min()))
    print('max patientid in test_y_encoded: ' +str(test_y_encoded.max()))

    num_of_people_in_data=np.unique(testY).shape[0]
    print('num_of_people_in_data: '+str(num_of_people_in_data))

    # Create the Keras Model
    keras_model = network_spatiotempo( num_of_people_in_data,learning_rate=args.learning_rate)
    callbacks = [EarlyStopping(monitor='val_loss', patience=6),ModelCheckpoint(filepath='best_model_100k_identification.h5', monitor='val_loss', save_best_only=True)]
    history=keras_model.fit(trainX, train_y_encoded,epochs=200,callbacks=callbacks, batch_size=1500,validation_data=(testX,test_y_encoded))
    # model.load_weights('best_model_100k_identification.h5')

    export_path = os.path.join(args.job_dir, 'keras_export')
    tf.keras.experimental.export_saved_model(keras_model, export_path)
    print('Model exported to: {}'.format(export_path))

    prediction_proba=keras_model.predict(testX)
    y_pred=np.argmax(prediction_proba,axis=1)
    y_pred= y_pred.reshape(y_pred.shape[0],1)
    test_y_encoded=test_y_encoded.reshape(test_y_encoded.shape[0],1)
    print('y_pred shape:'+str(y_pred.shape))
    print('test_y_encoded shape:'+str(test_y_encoded.shape))
    print('heartbeat_attributes_test_all shape:'+str(attributes_all_test.shape))

    test_results_attributes=np.hstack(( y_pred,test_y_encoded,attributes_all_test))
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%Y-%m-%d-%H-%M-%S")

    dest = 'gs://ecg-data/test-results/iden-100k/identification_results_attributes_100k-'+timestampStr+'.csv'
    np.savetxt(file_io.FileIO(dest, 'w'), test_results_attributes)

    history_dest = 'gs://ecg-data/test-results/iden-100k/model_training-history-'+timestampStr+'.json'
    history_dict = history.history
    json.dump(str(history_dict), file_io.FileIO(history_dest, 'w'))


if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)
