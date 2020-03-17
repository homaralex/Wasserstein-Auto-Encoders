from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    for filename in (
        'results_download/Wasserstein-Auto-Encoders/experiments_old/celebA_256/col_dec/lmbd_31.62_run_1',
        'results_download/Wasserstein-Auto-Encoders/experiments_old/celebA_256/col_dec/lmbd_31.62_run_2',
        'results_download/Wasserstein-Auto-Encoders/experiments_old/celebA_256/col_dec/lmbd_100_run_1',
        'results_download/Wasserstein-Auto-Encoders/experiments/celebA_256/random/lmbd_10_run_5',
        'results_download/Wasserstein-Auto-Encoders/experiments/celebA_256/random/lmbd_.1_run_5',
    ):
        path = Path(filename) / 'loss_train.log'
        lines = path.read_text().split('\n')

        rows = []
        for line in lines:
            row = {
                val.rsplit(' ', 2)[0].strip(): float(val.rsplit(' ', 2)[1]) for val in line.split('\t')[:-1]
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        for y_name in ('Logvar penalty loss:', 'Regulariser loss:'):
            df.plot.scatter(
                x='Iteration',
                y=y_name,
                label=path.parent.name,
            )
            plt.show()
