import argparse

import numpy as np

import utils
from disentanglement_metric import Disentanglement


class DisentanglementAnalysis(Disentanglement):
    def __init__(self, model):
        self.model = model
        self.imgs, self.latents_sizes, self.latents_bases = utils.load_disentanglement_data_dsprites()

    def run(
            self,
            batch_size=30,
            num_batches=100,
            num_factors=5,
            mean_reprs=True,
    ):
        reprs_per_factor = []
        for factor_id in range(num_factors):
            diff_reprs = []
            for _ in range(num_batches):
                # sample batches of ground-truth factors
                c1 = self.sample_latent(batch_size)
                c2 = self.sample_latent(batch_size)

                # set factor_id pairwise equal in both batches
                c2[:, -factor_id] = c1[:, -factor_id]

                # fetch corresponding images from the dataset
                x1 = self.imgs[self.latent_to_index(c1)][:, :, :, None]
                x2 = self.imgs[self.latent_to_index(c2)][:, :, :, None]

                z1 = self.model.encode(x1, mean=mean_reprs)
                z2 = self.model.encode(x2, mean=mean_reprs)

                # compute the pairwise differences
                z_diff = np.abs(z1 - z2)
                diff_reprs.append(z_diff)

            reprs_per_factor.append(diff_reprs)

        reprs_per_factor = np.array(reprs_per_factor).reshape(num_factors, -1, self.model.z_dim)
        mean_reprs_per_factor = reprs_per_factor.mean(axis=1)

        num_dims_per_factor = 3
        # print the dimensions with the lowest variance
        least_var_dims = mean_reprs_per_factor.argsort(axis=1)[:, :num_dims_per_factor]
        print(least_var_dims)
        print(np.sort(mean_reprs_per_factor, axis=1)[:, :num_dims_per_factor].round(3))

        # then the ones with the highest
        # print(mean_reprs_per_factor.argsort(axis=1)[:, -num_dims_per_factor:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-e', '--experiment_path', type=str, required=True)
    args = parser.parse_args()

    model = utils.load_model(experiment_path=args.experiment_path)
    analysis = DisentanglementAnalysis(model)

    analysis.run(num_batches=1000)
    analysis.run(num_batches=1000, mean_reprs=False)
