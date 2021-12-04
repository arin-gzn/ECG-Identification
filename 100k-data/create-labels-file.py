import pandas as pd
import sys
from sklearn import preprocessing
from google.cloud import storage


start_number_per_dataset={'china_private':0,'china2':1000000,'usa_georgia':200000,'germany':300000}

def extract_labels(dataset_name,race,file_name,bucket_name,file_path):
    try:
        age=None
        sex=None
        diagnosis=None

        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.get_blob(file_path)
        downloaded_blob = str(blob.download_as_string())
        label_file_lines=downloaded_blob.replace('\\r','').split('\\n')

        for line in label_file_lines:
            line_cleaned=line.strip().lower()
            if line_cleaned.startswith('#age:'):
                age=line_cleaned[5:].strip()
                if age=='nan' or age=='' or age=='NULLValue':
                    age=-1
            elif line_cleaned.startswith('#sex:'):
                sex=line_cleaned[5:].strip()
                if sex=='f' or sex=='female':
                    sex='F'
                elif sex=='m' or sex=='male':
                    sex='M'
                elif sex=='':
                    sex='U'
                else:
                    sex='U'
            elif line_cleaned.startswith('#dx:'):
                diagnosis=line_cleaned[4:].strip()
                if diagnosis=='':
                    diagnosis='UNSPECIFIED'

        file_prefix=file_name.split('.')[0]
        if '_' in file_prefix:
            name_splitted=file_prefix.split('_')
            patient_id=name_splitted[0]
            sample_num=name_splitted[1]
        else:
            patient_id=file_prefix
            sample_num=0

        return {'dataset_name':dataset_name,'patient_id':patient_id,'file_name':file_prefix+'.csv','age':age,'gender':sex,'diagnosis':diagnosis,'race':race,'sample_num':sample_num}
    except Exception as e:
        print(str(e))
        print(file_path)
        return None



def create_labels_csv_from_headers_dir(dataset_name,race,bucket_name,gs_dir,output_csv_file_path):
    labels_list=[]
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    files_processed=0
    for blob in bucket.list_blobs(prefix=gs_dir):
        if blob.name.endswith(".hea"):
            files_processed=files_processed+1
            if files_processed % 100 == 0:
                print(files_processed, 'done!')
            labels=extract_labels(dataset_name,race,blob.name.rsplit('/', 1)[-1],bucket_name,blob.name)
            if labels is not None:
                labels_list.append(labels)

    labels_df=pd.DataFrame(labels_list)
    labels_df['race_encoded']=labels_df['race'].map({'chinese': 0, 'german': 1, 'american':2})
    labels_df['gender_encoded']=labels_df['gender'].map({'M': 0, 'F': 1, 'U':2})

    le = preprocessing.LabelEncoder()
    le.fit(labels_df.diagnosis)
    labels_df['diagnosis_encoded'] = le.transform(labels_df.diagnosis)
    labels_df['diagnosis_encoded'] = labels_df.apply(lambda x: start_number_per_dataset[x['dataset_name']]+x['diagnosis_encoded'], axis=1)

    le.fit(labels_df.patient_id)
    labels_df['patient_id_encoded'] = le.transform(labels_df.patient_id)
    labels_df['patient_id_encoded'] = labels_df.apply(lambda x: start_number_per_dataset[x['dataset_name']]+x['patient_id_encoded'], axis=1)


    labels_df.to_csv(output_csv_file_path, index=False)
    print(labels_df.head())




if __name__=='__main__':
    bucket_name=sys.argv[1]
    gs_dir=sys.argv[2]
    output_csv=sys.argv[3]
    race=sys.argv[4]
    dataset_name=sys.argv[5]

    # bucket_name=ecg-data
    # gs_dir=100k-data/PTBData_Germany/Headers
    # output_csv=german-labels.csv
    # race=german
    # dataset_name=germany

    # bucket_name=ecg-data
    # gs_dir=100k-data/CSPCData_China/Header
    # output_csv=china2-labels.csv
    # race=chinese
    # dataset_name=china2
    #
    # bucket_name=ecg-data
    # gs_dir=100k-data/GeorgiaData_USA/Header
    # output_csv=usa-georgia-labels.csv
    # race=american
    # dataset_name=usa_georgia

    # python3 -u create-labels-file.py $bucket_name $gs_dir $output_csv $race $dataset_name

    create_labels_csv_from_headers_dir(dataset_name,race,bucket_name,gs_dir,output_csv)


