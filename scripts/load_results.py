import pickle
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

DOWNLOAD_PATH = Path('results_download')
OPTS_FILENAME = 'opts.pickle'
FID_FILENAME = 'test_fid.txt'
TEST_ERROR_FILENAME = 'test_error.txt'
NUM_ACTIVE_DIMS_FILENAME = 'num_active_dims.txt'
DISENTANGLEMENT_4_FILENAME = 'disentanglement4.txt'
DISENTANGLEMENT_5_FILENAME = 'disentanglement5.txt'
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

    fid_score_file = test_error_file.parent / FID_FILENAME
    if fid_score_file.exists():
        opts_dict['test_fid_score'] = float(fid_score_file.read_text())

    num_active_dims_file = test_error_file.parent / NUM_ACTIVE_DIMS_FILENAME
    if num_active_dims_file.exists():
        opts_dict['num_active_dims'] = float(num_active_dims_file.read_text())

    for disentanglement_filename in (DISENTANGLEMENT_4_FILENAME, DISENTANGLEMENT_5_FILENAME):
        disentanglement_file = test_error_file.parent / disentanglement_filename
        if disentanglement_file.exists():
            for idx, dis_score in enumerate(disentanglement_file.read_text().split('\n')[3:]):
                opts_dict[disentanglement_filename.replace('.txt', '')] = float(dis_score)

    # omit zero values due to log-scaling of the x-axis in the plots
    if opts_dict['lambda_logvar_regularisation'] == 0:
        opts_dict['lambda_logvar_regularisation'] = 1e-4

    rows.append(opts_dict)

df = pd.DataFrame(rows)
metrics = (
    'test_rec_error',
    'test_fid_score',
    'num_active_dims',
    'disentanglement4',
    'disentanglement5',
)
# add metrics manually if they are not found in the data (so that all further code can be run)
for metric in metrics:
    if metric not in df:
        df[metric] = np.nan

# plot FID vs rec. error
# filter out runs without any regularisation
grouped = df.loc[df.lambda_logvar_regularisation > 1e-4].groupby([
    'z_dim',
    'z_logvar_regularisation',
])
ncols = df.z_logvar_regularisation.nunique()
nrows = df.z_dim.nunique()

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8), sharey='row', sharex='row')

for metric in ('test_fid_score', 'disentanglement4', 'disentanglement5'):
    for (key, ax) in zip(sorted(itertools.product(df.z_dim.unique(), df.z_logvar_regularisation.unique())),
                         axes.flatten()):
        try:
            subplot = grouped.get_group(key).plot.scatter(
                x='test_rec_error',
                y=metric,
                label=key,
                ax=ax,
            )
            ax.legend(loc='upper left')
        except KeyError:
            print(f'Group {key} not present')
    plt.savefig(PLOTS_DIR / f'{metric}_vs_rec_error.png')
    plt.show()

# plot all metrics vs reg. strength
grouped = df.groupby([
    'z_dim',
    'lambda_logvar_regularisation',
    'z_logvar_regularisation',
])

print(grouped.test_rec_error.mean())
print(grouped.test_fid_score.mean())
print(grouped.num_active_dims.mean())
print(grouped.disentanglement4.mean())
print(grouped.disentanglement5.mean())

for use_orig_scale in (False,):  # (True, False):
    for all_methods in (True, False):
        for metric in metrics:
            sub_df = (df if all_methods else df.loc[df.z_logvar_regularisation.isin(('L1', 'col_L1_enc'))])
            dims_and_methods = sorted(list(itertools.product(
                sub_df.z_dim.unique(),
                sub_df.z_logvar_regularisation.unique()),
            ))
            grouped = sub_df.groupby([
                'z_dim',
                'z_logvar_regularisation',
            ])

            ncols = df.z_logvar_regularisation.nunique() if all_methods else 2
            nrows = df.z_dim.nunique()

            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8), sharey='row')

            for (key, ax) in zip(dims_and_methods, axes.flatten()):
                try:
                    subplot = grouped.get_group(key).plot.scatter(
                        x='lambda_logvar_regularisation',
                        y=metric,
                        logx=True,
                        label=key,
                        ax=ax,
                    )
                    ax.legend(loc='upper left')
                    max_x_val = grouped.lambda_logvar_regularisation.max().max()
                    ax.set_xlim((5e-5, 1.5 * grouped.lambda_logvar_regularisation.max().max()))
                    if use_orig_scale:
                        if 'rec' in metric:
                            ax.set_ylim((6350, 6550) if key[0] == 32 else (6250, 6450))
                        else:
                            ax.set_ylim((70, 90) if key[0] == 32 else (70, 150))
                except KeyError:
                    print(f'Group {key} not present')

            plt.savefig(
                PLOTS_DIR / f'{metric}_{"orig" if use_orig_scale else "auto"}_scale{"_all" if all_methods else ""}.png')
            plt.show()
