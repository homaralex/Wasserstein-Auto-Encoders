CHECKPOINT_PATHS=$(find $1 -type f -name 'checkpoint')
for CHECKPOINT_PATH in $CHECKPOINT_PATHS; do
  EXPERIMENT_PATH=$(dirname $(dirname $CHECKPOINT_PATH))
  DISENTANGLEMENT_SCORES_PATH=${EXPERIMENT_PATH}/disentanglement_metrics.json
  if [ -f ${DISENTANGLEMENT_SCORES_PATH} ]; then
    echo "${DISENTANGLEMENT_SCORES_PATH} exists - skipping directory"
  else
    python disentanglement_metrics.py --experiment_path $EXPERIMENT_PATH
  fi
done
