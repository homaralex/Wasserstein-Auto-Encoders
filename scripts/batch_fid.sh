CHECKPOINT_PATHS=$(find $1 -type f -name 'checkpoint')
for CHECKPOINT_PATH in $CHECKPOINT_PATHS; do
  python fid.py --experiment_path $(dirname $(dirname $CHECKPOINT_PATH))
done