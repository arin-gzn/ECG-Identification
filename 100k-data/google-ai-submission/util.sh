
#gcloud ai-platform local train   --package-path trainer/   --module-name trainer.task    --job-dir ./




BUCKET=ecg-data
JOB_NAME="ecg_age_100k_spatiotempo_10dropout_run102"
JOB_DIR="gs://$BUCKET/keras-models/$JOB_NAME"
PROJECT=$PROJECT_ID
REGION=us-west1

gcloud ai-platform jobs submit training $JOB_NAME \
--package-path trainer/ \
--module-name trainer.train-age \
--region $REGION \
--python-version 3.7 \
--runtime-version 1.15 \
--job-dir $JOB_DIR \
--stream-logs \
--config config.yml



