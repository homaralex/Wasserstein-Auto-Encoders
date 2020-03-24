for i in {1..2}; do
  for penalty in "row" "col"; do
    for z_dim in 32 256; do
      for lmbd in ".1" "1" "5" "10" "100"; do
        python run.py --experiment celebA_random_col_dec --z_logvar_regularisation ${penalty}_L1_dec_proximal --lambda_logvar_regularisation $lmbd --z_dim $z_dim --experiment_path experiments/celebA_${z_dim}/${penalty}_dec_proximal/lmbd_${lmbd}_run_${i}
      done
    done
  done
done