from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from io import BytesIO
from tensorflow.python.lib.io import file_io
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from tensorflow.python.keras.layers import Input, Activation,Dense, Convolution2D,MaxPool2D,Flatten,BatchNormalization,Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


import argparse
import os
import numpy as np

import tensorflow as tf
from sklearn import preprocessing
from google.cloud import storage



def network(num_of_people_in_data,learning_rate=0.01):
    path='gs://ecg-data/ecg_iden_1data_chinaprivate_originalarchitecture_run2/keras_export'
    model = tf.keras.experimental.load_from_saved_model(path)
    model.summary()

    # add new  layers
    x = model.layers[-4].output
    x = Dense(32, activation='relu', name='transfer-dense')(x)
    x = Dense(16, activation='relu', name='transfer-dense2')(x)
    main_output = Dense(num_of_people_in_data, activation='softmax', name='transefer_output')(x)
    model = Model(inputs=model.inputs, outputs=main_output)
    for layer in model.layers[:-3]:
        layer.trainable = False
    for layer in model.layers[-3:]:
        layer.trainable=True
        print('trainable layer: '+str(layer.name))
    # learning_rate=5e-5
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics = ['sparse_categorical_accuracy'])
    print('New model')
    model.summary()

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





def train_and_evaluate(args):


    all_data_loaded=load_np_array_from_gs_dirs('ecg-data',['100k-data/china_private1/repeateds/train' ])
    all_data_nparr=all_data_loaded[~np.any(np.isnan(all_data_loaded) , axis=1)]
    all_data_nparr = all_data_nparr.reshape((all_data_nparr.shape[0], 12, 305))
    trainX=all_data_nparr[:,:,:300].reshape((-1, 12, 300, 1))
    trainY=all_data_nparr[:,0,300]
    print('trainY size: '+str(trainY.shape))
    print('trainY size: '+str(trainY.shape))


    all_data_loaded=load_np_array_from_gs_dirs('ecg-data',['100k-data/china_private1/repeateds/test' ])
    all_data_nparr=all_data_loaded[~np.any(np.isnan(all_data_loaded) , axis=1)]
    all_data_nparr = all_data_nparr.reshape((all_data_nparr.shape[0], 12, 305))
    testX=all_data_nparr[:,:,:300].reshape((-1, 12, 300, 1))
    testY=all_data_nparr[:,0,300]
    print('testX size: '+str(testX.shape))
    print('testY size: '+str(testY.shape))



    le = preprocessing.LabelEncoder()
    le.fit(np.concatenate((trainY,testY)))
    train_y_encoded = le.transform(trainY)
    test_y_encoded = le.transform(testY)

    num_of_people_in_data=np.unique(testY).shape[0]
    print('test_df size: '+str(num_of_people_in_data))

    # Create the Keras Model
    keras_model = network( num_of_people_in_data,learning_rate=args.learning_rate)
    callbacks = [EarlyStopping(monitor='val_loss', patience=8),ModelCheckpoint(filepath='best_model_100k_identification.h5', monitor='val_loss', save_best_only=True)]
    keras_model.fit(trainX, train_y_encoded,epochs=200,callbacks=callbacks, batch_size=500,validation_data=(testX,test_y_encoded))
    # model.load_weights('best_model_100k_identification.h5')

    export_path = os.path.join(args.job_dir, 'keras_export')
    tf.keras.experimental.export_saved_model(keras_model, export_path)
    print('Model exported to: {}'.format(export_path))


if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)
