for i in {1..5}; do
  for z_dim in 32 256; do
    for lmbd in "0" ".001" ".01" ".1" "1"; do
      python run.py --experiment celebA_random --z_logvar_regularisation L1 --lambda_logvar_regularisation $lmbd --z_dim $z_dim --experiment_path experiments/celebA_${z_dim}/random/lmbd_${lmbd}_run_${i}
    done
  done
done