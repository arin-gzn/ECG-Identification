PROJECT=$PROJECT_ID
REGION=us-west1
BUCKET=ecg-data
LABELS_INPUT_FILE=gs://ecg-data/small-data-germany/german-labels.csv
ECG_DATA_DIR=gs://ecg-data/100k-data/PTBData_Germany/DenData
OUTPUT=gs://ecg-data/small-data-germany/output

python preprocess_dataflow.py \
  --service_account=$ACCOUNT\
  --region $REGION \
  --runner DataflowRunner \
  --project $PROJECT \
  --temp_location gs://$BUCKET/tmp/ \
  --num_workers 3 \
  --max_num_workers 5 \
  --labels-file-path $LABELS_INPUT_FILE \
  --ecg-data-dir-input $OUTPUT \
  --output $OUTPUT \
  --setup_file=./setup.py



 gcloud auth activate-service-account dataflow-service@trans-proposal-217500.iam.gserviceaccount.com  --key-file=/Users/aring/dataflow-key.json --project=trans-proposal-217500




INPUT=/Users/aring/IdeaProjects/ECG-biometric/src/100k-data/sample-data/germany/german-labels.csv
OUTPUT=/Users/aring/IdeaProjects/ECG-biometric/src/100k-data/sample-data/output-dataflow
INPUT1=/Users/aring/IdeaProjects/ECG-biometric/src/100k-data/sample-data/germany/DenData

python preprocess_dataflow.py --input_labels_file $INPUT --input_ecg_data_dir $INPUT1 --output $OUTPUT --runner DirectRunner


PROJECT_ID=trans-proposal-217500
BUCKET=ecg-data

python preprocess_dataflow.py     --runner DataflowRunner     --project $PROJECT_ID     --staging_location gs://$BUCKET/staging     --temp_location gs://$BUCKET/temp     --template_location gs://$BUCKET/templates/ECG_Preprocess --setup_file=./setup.py


gs://ecg-data/100k-data/GeorgiaData_USA/DenData
gs://ecg-data/100k-data/GeorgiaData_USA/german_labels.csv
gs://ecg-data/100k-data/GeorgiaData_USA/output/output