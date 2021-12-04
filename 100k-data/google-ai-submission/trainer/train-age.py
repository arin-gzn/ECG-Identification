from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.preprocessing import StandardScaler

from io import BytesIO
from tensorflow.python.lib.io import file_io
import numpy as np
from google.cloud import storage
from sklearn.model_selection import GroupShuffleSplit


from sklearn.model_selection import train_test_split

import pandas as pd

from tensorflow.python.keras.layers import Input, Dense, Activation,Dropout,Convolution2D,MaxPool2D,Flatten,BatchNormalization
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
    main_output = Dense(1, name='main_output')(dense_end2)
    model = Model(inputs= inputs_cnn, outputs=main_output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', 'mape'])

    return model



def network_spatiotempo(learning_rate=0.01):
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
    x=Dropout(0.1)(x)

    x = Dense(64, activation='relu')(x)
    x=Dropout(0.1)(x)

    main_output = Dense(1, name='main_output')(x)
    model = Model(inputs= inputs_cnn, outputs=main_output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', 'mape'])


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


    main_output = Dense(1, name='main_output')(x)
    model = Model(inputs= inputs_cnn, outputs=main_output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', 'mape'])

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



def train_and_evaluate_old(args):

    # all_data_loaded=load_np_array_from_gs_dirs('ecg-data',['100k-data/CSPCData_China/output' ,'100k-data/GeorgiaData_USA/output','100k-data/china_private1/output','100k-data/PTBData_Germany/output'])
    # ,'100k-data/GeorgiaData_USA/output','100k-data/china_private1/output','100k-data/PTBData_Germany/output'
    all_data_loaded=load_np_array_from_gs_dirs('ecg-data',['100k-data/china_private1/output' ])

    print('all data size: '+str(all_data_loaded.shape))
    all_data_loaded_df = pd.DataFrame(data=all_data_loaded)

    all_data_loaded_df=all_data_loaded_df[all_data_loaded_df.iloc[:,3657]!=-1]


    train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 42).split(all_data_loaded_df, groups=all_data_loaded_df.iloc[:,3655]))
    train_df = all_data_loaded_df.iloc[train_inds]
    test_df = all_data_loaded_df.iloc[test_inds]
    # train_df, test_df = train_test_split(all_data_loaded_df, test_size=0.2, random_state=42, shuffle=True)

    train_df_race_combination=train_df.groupby(train_df.columns[3657]).size().reset_index(name='count')
    print(train_df_race_combination.head())

    test_df_race_combination=test_df.groupby(test_df.columns[3657]).size().reset_index(name='count')
    print(test_df_race_combination.head())

    print('train_df size: '+str(train_df.shape))
    print('test_df size: '+str(test_df.shape))

    train_nparr=train_df.values
    test_nparr=test_df.values


    print('X  sample row: '+str(train_nparr[10]))
    print('X  last element: '+str(train_nparr[10,299]))
    print('y  sample row: '+str(test_nparr[10]))

    train_nparr = train_nparr.reshape((train_nparr.shape[0], 12, 305))
    test_nparr = test_nparr.reshape((test_nparr.shape[0], 12, 305))

    trainX=train_nparr[:,:,:300].reshape((-1, 12, 300, 1))
    trainY=train_nparr[:,0,302]

    testX=test_nparr[:,:,:300].reshape((-1, 12, 300, 1))
    testY=test_nparr[:,0,302]

    keras_model = network( learning_rate=args.learning_rate)
    callbacks = [EarlyStopping(monitor='val_loss', patience=8),ModelCheckpoint(filepath='best_model_age_100k.h5', monitor='val_loss', save_best_only=True)]
    keras_model.fit(trainX, trainY,epochs=100,callbacks=callbacks, batch_size=2000,validation_data=(testX,testY))

    export_path = os.path.join(args.job_dir, 'keras_export')
    tf.keras.experimental.export_saved_model(keras_model, export_path)
    print('Model exported to: {}'.format(export_path))



def add_str(x,my_prefix):
    return "{}-{}".format(x, my_prefix)



def train_and_evaluate_100k(args):
    # X_scaled_cspchina,y_cspchina,attributes_all_cspchina=load_and_scale('ecg-data',['100k-data/CSPCData_China/output-small' ],0)
    # X_scaled_georgiausa,y_georgiausa,attributes_all_georgiausa=load_and_scale('ecg-data',['100k-data/GeorgiaData_USA/output-small' ],1)
    # X_scaled_china_private1,y_china_private1,attributes_all_china_private1=load_and_scale('ecg-data',['100k-data/china_private1/output-small' ],2)
    # X_scaled_germany,y_germany,attributes_all_germany=load_and_scale('ecg-data',['100k-data/PTBData_Germany/output-small' ],3)


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

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)

    attributes_all=np.concatenate(( attributes_all_georgiausa,attributes_all_china_private1,attributes_all_cspchina,attributes_all_germany), axis=0)
    attributes_all_georgiausa=None
    attributes_all_china_private1=None
    attributes_all_cspchina=None
    attributes_all_germany=None

    X=X.reshape((X.shape[0],12*300))
    y_encoded=y_encoded.reshape((y_encoded.shape[0],1))

    print('X shape'+str(X.shape))
    print('y_encoded shape'+str(y_encoded.shape))
    print('attributes_all shape'+str(attributes_all.shape))


    all_data_loaded_df=np.hstack(( X,y_encoded,attributes_all))
    all_data_loaded_df = pd.DataFrame(data=all_data_loaded_df)
    all_data_loaded_df=all_data_loaded_df[all_data_loaded_df.iloc[:,3603]!=-1]

    print('all_data_loaded_df shape'+str(all_data_loaded_df.shape))





    train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 100).split(all_data_loaded_df, groups=all_data_loaded_df.iloc[:,3600]))
    train_df = all_data_loaded_df.iloc[train_inds]
    test_df = all_data_loaded_df.iloc[test_inds]

    print('train_df unique patient ids: '+str(train_df.iloc[:,3600].unique().size))
    print('test_df unique patient ids: '+str(test_df.iloc[:,3600].unique().size))

    train_df_race_combination=train_df.groupby(train_df.columns[3603]).size().reset_index(name='count')
    print(train_df_race_combination.head())

    test_df_race_combination=test_df.groupby(test_df.columns[3603]).size().reset_index(name='count')
    print(test_df_race_combination.head())

    print('train_df size: '+str(train_df.shape))
    print('test_df size: '+str(test_df.shape))

    train_nparr=train_df.values
    test_nparr=test_df.values


    trainX=train_nparr[:,:3600].reshape((-1, 12, 300,1))
    trainY=train_nparr[:,3603]

    testX=test_nparr[:,:3600].reshape((-1, 12, 300,1))
    testY=test_nparr[:,3603]



    print('unique values in testY: '+str(len(np.unique(testY)) ))
    print('unique values in trainY: '+str(len(np.unique(trainY)) ))

    print('X size: '+str(train_nparr.shape))
    print('trainX size: '+str(trainX.shape))
    print('testX size: '+str(testX.shape))


    keras_model = network_spatiotempo( learning_rate=args.learning_rate)
    callbacks = [EarlyStopping(monitor='val_loss', patience=6),ModelCheckpoint(filepath='best_model_gender_100k.h5', monitor='val_loss', save_best_only=True)]
    keras_model.fit(trainX, trainY,epochs=200,callbacks=callbacks, batch_size=2000,validation_data=(testX,testY))

    export_path = os.path.join(args.job_dir, 'keras_export')
    tf.keras.experimental.export_saved_model(keras_model, export_path)
    print('Model exported to: {}'.format(export_path))






if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate_100k(args)
