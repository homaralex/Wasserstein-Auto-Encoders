# this script is used to download results from experiments where test reconstruction error is measured
# at the end of training (so far only CelebA)
# first argument is the credentials to log in to the ssh server

# file name with per-experiment configuration dictionary
OPTS_FILENAME='opts.pickle'
# file name for FID scores
FID_FILENAME='test_fid.txt'
# file name for active dims
ACTIVE_DIMS_FILENAME='num_active_dims.txt'
# directory to download the files to on the local machine
DOWNLOAD_PATH='results_download'
mkdir $DOWNLOAD_PATH

TEST_ERROR_PATHS=$(ssh $1 find "Wasserstein-Auto-Encoders/experiments/" -type f -name 'test_error.txt')
for TEST_ERROR_PATH in $TEST_ERROR_PATHS; do
  # create experiment directory tree on local machine
  mkdir -p ${DOWNLOAD_PATH}/$(dirname $TEST_ERROR_PATH)
  echo "Downloading ${TEST_ERROR_PATH}"
  # download file with test_error
  rsync ${1}:${TEST_ERROR_PATH} ${DOWNLOAD_PATH}/${TEST_ERROR_PATH}
  # create path to the opts file and download it too
  OPTS_PATH=$(dirname $TEST_ERROR_PATH)/${OPTS_FILENAME}
  rsync ${1}:${OPTS_PATH} ${DOWNLOAD_PATH}/${OPTS_PATH}
  # download fid score file
  FID_PATH=$(dirname $TEST_ERROR_PATH)/${FID_FILENAME}
  rsync ${1}:${FID_PATH} ${DOWNLOAD_PATH}/${FID_PATH}
  # download active dims file
  ACTIVE_DIMS_PATH=$(dirname $TEST_ERROR_PATH)/${ACTIVE_DIMS_FILENAME}
  rsync ${1}:${ACTIVE_DIMS_PATH} ${DOWNLOAD_PATH}/${ACTIVE_DIMS_PATH}
done
