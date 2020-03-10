import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DOWNLOAD_PATH = Path('results_download')
OPTS_FILENAME = 'opts.pickle'
TEST_ERROR_FILENAME = 'test_error.txt'
PLOTS_DIR = Path('plots')
PLOTS_DIR.mkdir(exist_ok=True)

rows = []
for test_error_file in DOWNLOAD_PATH.glob(f'**/{TEST_ERROR_FILENAME}'):
    opts_file = test_error_file.parent / OPTS_FILENAME
    assert opts_file.exists(), f'Could not find corresponding opts file for the experiment:\n{test_error_file.parent}'

    # read the test reconstruction error and cast it to float
    test_error = float(test_error_file.read_text())
    # load the pickle with experiment configuration
    opts_dict = pickle.load(opts_file.open('rb'))
    opts_dict['test_rec_error'] = test_error

    rows.append(opts_dict)

df = pd.DataFrame(rows)
grouped = df.groupby([
    'z_dim',
    'z_logvar_regularisation',
])
print(grouped.test_rec_error.mean())

ncols = df.z_logvar_regularisation.nunique()
nrows = int(np.ceil(grouped.ngroups / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8), sharey='row')

for use_orig_scale in [True, False]:
    for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
        subplot = grouped.get_group(key).plot.scatter(
            x='lambda_logvar_regularisation',
            y='test_rec_error',
            logx=True,
            label=key,
            ax=ax,
        )
        ax.set_xlim((.0005, 1.5))
        if use_orig_scale:
            ax.set_ylim((6350, 6550) if key[0] == 32 else (6250, 6450))

    plt.savefig(PLOTS_DIR / f'preliminary_{"orig" if use_orig_scale else "auto"}_scale.png')
    plt.show()