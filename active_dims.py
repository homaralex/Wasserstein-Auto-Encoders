import argparse
import pickle

import numpy as np

import wae


def load_model(experiment_path, dataset=None):
    with open(experiment_path + "/opts.pickle", 'rb') as f:
        opts = pickle.load(f)
    if dataset is not None:
        opts['dataset'] = dataset
    model = wae.Model(opts, load=True)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-n', '--num_samples', type=int, default=10000)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-e', '--experiment_path', type=str, required=True)
    parser.add_argument('-d', '--dataset', type=str, default=None)
    args = parser.parse_args()

    model = load_model(experiment_path=args.experiment_path, dataset=args.dataset)
    log_vars = model.get_variances(num_samples=args.num_samples, batch_size=args.batch_size)
    mean_vars = np.exp(log_vars.mean(axis=0))
    num_active_dims = (1 - mean_vars).sum().round(2)

    # load_model changes cwd to experiment_path (in wae.Model init) so we can save the results here simply
    with open('num_active_dims.txt', 'w') as out_file:
        out_file.write(str(num_active_dims))
    print(num_active_dims)
