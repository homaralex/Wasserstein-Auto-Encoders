import argparse
from pathlib import Path

import gin
import disentanglement_lib.evaluation.metrics.mig as dlib_mig

import utils
from disentanglement_metric import Disentanglement

METRICS = ('mig',)
# use absolute path as WAE.__init__ changes current dir
METRIC_CONFIGS_DIR = Path('metric_configs').absolute()
METRIC_CONFIGS_PATHS = {metric_name: METRIC_CONFIGS_DIR / f'{metric_name}.gin' for metric_name in METRICS}


class OtherDisentanglement(Disentanglement):
    def _get_data_points(self, num_points):
        ys = self.sample_latent(size=num_points)
        xs = self.imgs[self.latent_to_index(ys)][:, :, :, None]

        return xs, ys

    def run(self, num_points=1e4):
        # cast to int as scientific notation numbers are floats
        num_points = int(num_points)

        xs, ys = self._get_data_points(num_points=num_points)
        zs = self.model.encode(xs)

        config_path = METRIC_CONFIGS_PATHS['mig']
        gin.parse_config_file(config_file=str(config_path))

        mig = dlib_mig._compute_mig(
            # transpose as numpy expects num_factors as first dimension
            mus_train=zs.transpose(),
            # take only 5 last factors as the first one is a placeholder - which will result in 0 entropy, and thus
            # division by zero
            ys_train=ys.transpose()[1:],
        )

        return mig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-e', '--experiment_path', type=str, required=True)
    args = parser.parse_args()

    model = utils.load_model(experiment_path=args.experiment_path)
    metrics = OtherDisentanglement(model)

    mig = metrics.run(num_points=1e4)
    print(mig)
