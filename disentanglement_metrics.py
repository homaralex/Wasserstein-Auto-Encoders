import argparse
from pathlib import Path

import gin
import numpy as np
import disentanglement_lib.evaluation.metrics.utils as dlib_metrics_utils
import disentanglement_lib.evaluation.metrics.mig as dlib_mig
import disentanglement_lib.evaluation.metrics.dci as dlib_dci
import disentanglement_lib.evaluation.metrics.modularity_explicitness as dlib_modularity

import utils
from disentanglement_metric import Disentanglement

METRICS = ('mig', 'modularity_explicitness', 'dci')
# use absolute path as WAE.__init__ changes current dir
METRIC_CONFIGS_DIR = Path('metric_configs').absolute()
METRIC_CONFIGS_PATHS = {metric_name: METRIC_CONFIGS_DIR / f'{metric_name}.gin' for metric_name in METRICS}


class DisentanglementMetrics(Disentanglement):
    def _get_data_points(self, num_points):
        ys = self.sample_latent(size=num_points)
        xs = self.imgs[self.latent_to_index(ys)][:, :, :, None]

        zs = self.model.encode(xs)

        # transpose as numpy expects num_factors as first dimension
        # also, take only 5 last factors as the first one is a placeholder - which will result in 0 entropy, and thus
        # division by zero in some metrics
        zs, ys = zs.transpose(), ys.transpose()[1:]

        return zs, ys

    def mig(self, zs, ys):
        config_path = METRIC_CONFIGS_PATHS['mig']
        gin.parse_config_file(config_file=str(config_path))

        mig = dlib_mig._compute_mig(mus_train=zs, ys_train=ys)

        return mig

    def modularity(self, zs, ys):
        config_path = METRIC_CONFIGS_PATHS['modularity_explicitness']
        gin.parse_config_file(config_file=str(config_path))

        discretized_mus = dlib_metrics_utils.make_discretizer(zs)
        mutual_information = dlib_metrics_utils.discrete_mutual_info(discretized_mus, ys)

        return {'modularity_score': dlib_modularity.modularity(mutual_information)}

    def explicitness(self, zs, ys, zs_test, ys_test):
        config_path = METRIC_CONFIGS_PATHS['modularity_explicitness']
        gin.parse_config_file(config_file=str(config_path))

        mus_train, ys_train = zs, ys
        mus_test, ys_test = zs_test, ys_test

        scores = {}
        explicitness_score_train = np.zeros([ys_train.shape[0], 1])
        explicitness_score_test = np.zeros([ys_test.shape[0], 1])
        mus_train_norm, mean_mus, stddev_mus = dlib_metrics_utils.normalize_data(mus_train)
        mus_test_norm, _, _ = dlib_metrics_utils.normalize_data(mus_test, mean_mus, stddev_mus)
        for i in range(ys_train.shape[0]):
            explicitness_score_train[i], explicitness_score_test[i] = dlib_modularity.explicitness_per_factor(
                mus_train_norm, ys_train[i, :],
                mus_test_norm, ys_test[i, :],
            )
        scores["explicitness_score_train"] = np.mean(explicitness_score_train)
        scores["explicitness_score_test"] = np.mean(explicitness_score_test)

        return scores

    def dci(self, zs, ys, zs_test, ys_test):
        config_path = METRIC_CONFIGS_PATHS['dci']
        gin.parse_config_file(config_file=str(config_path))

        return dlib_dci._compute_dci(
            mus_train=zs,
            ys_train=ys,
            mus_test=zs_test,
            ys_test=ys_test,
        )

    def run(self, num_points=1e4):
        # cast to int as scientific notation numbers are floats
        num_points = int(num_points)

        zs, ys = self._get_data_points(num_points=num_points)
        zs_test, ys_test = self._get_data_points(num_points=num_points // 2)

        metrics = {}
        metrics.update(self.mig(zs=zs, ys=ys))
        metrics.update(self.modularity(zs=zs, ys=ys))
        metrics.update(self.explicitness(zs=zs, ys=ys, zs_test=zs_test, ys_test=ys_test))
        metrics.update(self.dci(zs=zs, ys=ys, zs_test=zs_test, ys_test=ys_test))

        return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-e', '--experiment_path', type=str, required=True)
    args = parser.parse_args()

    model = utils.load_model(experiment_path=args.experiment_path)
    disentanglement_metrics = DisentanglementMetrics(model)

    metrics = disentanglement_metrics.run(num_points=1e4)
    print(metrics)
