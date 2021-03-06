CHECKPOINT_PATHS=$(find $1 -type f -name 'checkpoint')
for CHECKPOINT_PATH in $CHECKPOINT_PATHS; do
  EXPERIMENT_PATH=$(dirname $(dirname $CHECKPOINT_PATH))
  ACTIVE_DIMS_PATH=${EXPERIMENT_PATH}/num_active_dims.txt
  if [ -f ${ACTIVE_DIMS_PATH} ]; then
    echo "${ACTIVE_DIMS_PATH} exists - skipping directory"
  else
    python active_dims.py --experiment_path $EXPERIMENT_PATH
  fi
done
