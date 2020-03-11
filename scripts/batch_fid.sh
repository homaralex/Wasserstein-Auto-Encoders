CHECKPOINT_PATHS=$(find $1 -type f -name 'checkpoint')
for CHECKPOINT_PATH in $CHECKPOINT_PATHS; do
  EXPERIMENT_PATH=$(dirname $(dirname $CHECKPOINT_PATH))
  FID_PATH=${EXPERIMENT_PATH}/test_fid.txt
  if [ ! -f ${FID_PATH} ]; then
    echo "${FID_PATH} exists - skipping directory"
  else
    python fid.py --experiment_path $EXPERIMENT_PATH
  fi
done
