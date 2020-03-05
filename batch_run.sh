for penalty in "dec" "enc" "enc_dec"; do
  for i in {1..5}; do
    for z_dim in 32 256; do
      for lmbd in ".001" ".01" ".1" "1"; do
        python run.py --experiment celebA_random_col_dec --z_logvar_regularisation col_L1_$penalty --lambda_logvar_regularisation $lmbd --z_dim $z_dim --experiment_path experiments/celebA_col_${penalty}/lmbd_${lmbd}_run_${i}
      done
    done
  done
done