PROJECT_ID=$PROJECT_ID
BUCKET=ecg-data

python preprocess_dataflow.py \
    --runner DataflowRunner \
    --project $PROJECT_ID \
    --staging_location gs://$BUCKET/staging \
    --temp_location gs://$BUCKET/temp \
    --template_location gs://$BUCKET/templates/ECG_Preprocess