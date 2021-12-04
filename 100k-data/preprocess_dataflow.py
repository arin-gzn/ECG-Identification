#!/usr/bin/env python3
import numpy as np
import csv
import io
from past.builtins import unicode
from scipy import signal

import argparse
import logging
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from io import StringIO
import os
import logging
import pandas as pd
from biosppy.signals import ecg
import sys
TARGET_SAMPLING_RATE=250
DATA_SAMPLING_RATE=500
INPUT_BEAT_SIZE=300

class process_12lead_ecg(beam.DoFn):
    def __init__(self, input_ecg_data_dir):
        self.input_ecg_data_dir = input_ecg_data_dir

    def process(self, csv_row):
        try:
            patinet_attributes_list=next(csv.reader(StringIO(csv_row), quotechar='"', delimiter=',',quoting=csv.QUOTE_MINIMAL, skipinitialspace=True))
            patient_id=patinet_attributes_list[11]
            age=patinet_attributes_list[3]
            gender=patinet_attributes_list[9]
            diagnosis=patinet_attributes_list[10]
            race=patinet_attributes_list[8]
            file_name=patinet_attributes_list[2]

            if file_name=='file_name':#header row
                return None


            path = self.input_ecg_data_dir.get()+ '/'+file_name
            all_beats_12lead_with_label_diagnosis = np.empty((0,12,INPUT_BEAT_SIZE+5), dtype=np.float)

            # TODO read from gs
            record = pd.read_csv(path, header=None)

            channel = record[1].values
            ecg_info_extracted = ecg.ecg(signal=channel, sampling_rate=DATA_SAMPLING_RATE, show=False)
            rpeaks=ecg_info_extracted['rpeaks']
            heart_beats_12lead =  np.empty((0,12,INPUT_BEAT_SIZE), dtype=np.float)
            for i in range((len(rpeaks)-1)):
                down_sampled_beat = signal.resample(record[:][rpeaks[i]:rpeaks[i+1]], 300).T
                heart_beats_12lead=np.vstack((heart_beats_12lead,np.reshape(down_sampled_beat,(1,12,INPUT_BEAT_SIZE))))

            num_of_beats=heart_beats_12lead.shape[0]

            for i in range(0,num_of_beats):
                one_beat_12lead = np.reshape(heart_beats_12lead[i], (12, 300))
                label_column = np.repeat(patient_id, one_beat_12lead.shape[0])
                label_column=np.reshape(label_column, (label_column.shape[0], 1))
                diagnosis_column=np.repeat(diagnosis, one_beat_12lead.shape[0])
                diagnosis_column=np.reshape(diagnosis_column, (diagnosis_column.shape[0], 1))
                age_column=np.repeat(age, one_beat_12lead.shape[0])
                age_column=np.reshape(age_column, (age_column.shape[0], 1))
                gender_column=np.repeat(gender, one_beat_12lead.shape[0])
                gender_column=np.reshape(gender_column, (gender_column.shape[0], 1))
                race_column=np.repeat(race, one_beat_12lead.shape[0])
                race_column=np.reshape(race_column, (race_column.shape[0], 1))
                one_beat_12lead_with_label = np.append(one_beat_12lead,label_column,1)
                one_beat_12lead_with_label = np.append(one_beat_12lead_with_label,diagnosis_column,1)
                one_beat_12lead_with_label = np.append(one_beat_12lead_with_label,age_column,1)
                one_beat_12lead_with_label = np.append(one_beat_12lead_with_label,gender_column,1)
                one_beat_12lead_with_label = np.append(one_beat_12lead_with_label,race_column,1)
                one_beat_12lead_with_label = np.reshape(one_beat_12lead_with_label,(1,12,305))
                all_beats_12lead_with_label_diagnosis=np.vstack((all_beats_12lead_with_label_diagnosis,one_beat_12lead_with_label))

        except Exception as e:
            logging.info('ERROR: file path',path )
            logging.info('ERROR: ',str(e) )
            return None
        yield all_beats_12lead_with_label_diagnosis


def save_to_numpyformat(mp_array):
    string_stream = io.BytesIO()
    np.savetxt(string_stream, mp_array.reshape((mp_array.shape[0],12*305)), delimiter=",", fmt='%s')
    return string_stream.getvalue().decode().rstrip()
    # np.savetxt('test.txt', ab, fmt="%10s %10.3f")

class ECGPreprocessOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_value_provider_argument(
            '--input_labels_file',
            dest='input_labels_file',
            help='raw Input file to process.')

        parser.add_value_provider_argument(
            '--input_ecg_data_dir',
            dest='input_ecg_data_dir',
            help='raw Input file to process.')

        parser.add_value_provider_argument(
            '--output',
            dest='output',
            required=False,
            help='Output file to write results to.')



def run(argv=None, save_main_session=True):
    # parser = argparse.ArgumentParser()
    # options = PipelineOptions(argv)
    # known_args, pipeline_args = parser.parse_known_args(argv)
    pipeline_options = PipelineOptions(argv)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    ECG_pipeline_options= pipeline_options.view_as(ECGPreprocessOptions)


    # pipeline_options = PipelineOptions(['--output', 'some/output_path'])
    # p = beam.Pipeline(options=pipeline_options)
    #
    # wordcount_options = pipeline_options.view_as(WordcountOptions)
    # lines = p | 'read' >> ReadFromText(wordcount_options.input)

    with beam.Pipeline(options=ECG_pipeline_options) as p:
        (p
         | 'Read labels file' >> ReadFromText(ECG_pipeline_options.input_labels_file).with_output_types(unicode)
         | 'process ecg' >> (beam.ParDo(process_12lead_ecg(ECG_pipeline_options.input_ecg_data_dir)))
         | 'Filter out None rows' >> (beam.Filter(lambda x: x is not None))
         | 'Filter out patients with less than 4 heartbeats' >> (beam.Filter(lambda x: x.shape[0]>4))
         | 'convert to numpy save format' >> (beam.Map(save_to_numpyformat))
         | 'Filter out None rows2' >> (beam.Filter(lambda x: x is not None))
         | 'WriteToText' >> beam.io.WriteToText(ECG_pipeline_options.output,file_name_suffix='.csv'))

    result = p.run()
    result.wait_until_finish()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()







