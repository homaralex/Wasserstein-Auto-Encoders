import wae
import argparse
import config

parser = argparse.ArgumentParser()
parser.add_argument("--experiment",
                    help='Default experiment configs to use: dsprites/fading_squares/celebA_mini/celebA_random/celebA_deterministic')
parser.add_argument("--dataset",
                    help='Dataset to train on: dsprites/celebA/celebA_mini/fading_squares')
parser.add_argument("--z_dim", help='latent space dimensionality', type=int)
parser.add_argument("--lambda_imq", help='Lambda for WAE penalty', type=float)
parser.add_argument("--experiment_path",
                    help="Relative path to where this experiment should save results")
parser.add_argument("--encoder_distribution",
                    help="Encoder distribution: deterministic/gaussian/uniform")
parser.add_argument("--z_prior",
                    help="Prior distribution over latent space: gaussian/uniform")
parser.add_argument("--loss_reconstruction",
                    help="Image reconstruction loss: bernoulli/L2_squared")
parser.add_argument("--loss_regulariser",
                    help="Model type: VAE/beta_VAE/WAE_MMD")
parser.add_argument("--beta", type=float,
                    help="beta parameter for beta_VAE")
parser.add_argument("--disentanglement_metric", type=bool,
                    help="Calculate disentanglement metric")
parser.add_argument("--make_pictures_every", type=int,
                    help="How often to plot random samples and reconstructions")
parser.add_argument("--save_every", type=int,
                    help="How often to save the model")
parser.add_argument("--batch_size", type=int,
                    help="Batch size. Default 100")
parser.add_argument("--encoder_architecture",
                    help="Architecture of encoder: FC_dsprites/small_convolutional_celebA")
parser.add_argument("--decoder_architecture",
                    help="Architecture of decoder: FC_dsprites/small_convolutional_celebA")
parser.add_argument("--z_logvar_regularisation",
                    help="Regularisation on log-variances: None/L1/L2_squared")
parser.add_argument("--lambda_logvar_regularisation", type=float,
                    help="Coefficient of logvariance regularisation")
parser.add_argument("--plot_losses",
                    help="Plot losses and least-gaussian-subspace: True/False:")
parser.add_argument("--adversarial_cost_n_filters", type=int,
                    help="Number of convolutional filters to use for adversarial cost")
parser.add_argument("--adv_cost_nlayers", type=int,
                    help="Number of convolutional layers to use for adversarial cost")
parser.add_argument("--adversarial_cost_kernel_size", type=int,
                    help="Size of convolutional kernels to use for adversarial cost. -1 for sum over kernels of size 3,4,5")
parser.add_argument("--adv_cost_lambda", type=float,
                    help="Weighting of adversarial cost")
parser.add_argument("--adv_cost_normalise_filter", type=bool,
                    help="Whether to normalise adversarial cost across filters (default uses Sylvain normalisation across channels)")
parser.add_argument("--pixel_wise_l2", type=bool,
                    help="Should mean pixel loss be over individual pixels or patches for patch_moments?")
parser.add_argument("--encoder_num_filters", type=int,
                    help="Number of filters for the encoder")
parser.add_argument("--decoder_num_filters", type=int,
                    help="Number of filters for the decoder")
parser.add_argument("--encoder_num_layers", type=int,
                    help="Number of layers for the encoder")
parser.add_argument("--decoder_num_layers", type=int,
                    help="Number of layers for the decoder")
parser.add_argument("--l2_lambda", type=float,
                    help="Weighting of l2 penalty")
parser.add_argument("--patch_classifier_lambda", type=float,
                    help="Weighting of the patch classification penalty")


FLAGS = parser.parse_args()

if __name__ == "__main__":
    if FLAGS.experiment == 'dsprites':
        opts = config.dsprites_opts
    elif FLAGS.experiment == 'fading_squares':
        opts = config.fading_squares_opts
    elif FLAGS.experiment == 'celebA_random':
        opts = config.celebA_random_opts
    elif FLAGS.experiment == 'celebA_random_col_dec':
        opts = config.celebA_random_col_dec_opts
    elif FLAGS.experiment == 'celebA_deterministic':
        opts = config.celebA_deterministic_opts
    elif FLAGS.experiment == 'celebA_mini':
        opts = config.celebA_mini_opts
    elif FLAGS.experiment == 'celebA_dcgan_deterministic':
        opts = config.celebA_dcgan_deterministic_opts
    elif FLAGS.experiment == 'grassli_VAE':
        opts = config.grassli_VAE_opts
    elif FLAGS.experiment == 'grassli_WAE':
        opts = config.grassli_WAE_opts
    elif FLAGS.experiment == 'celebA_dcgan_adv':
        opts = config.celebA_dcgan_adv_cost_opts
    elif FLAGS.experiment == 'celebA_dcgan_adv_l2_filters':
        opts = config.celebA_dcgan_adv_cost_l2_filters_opts
    elif FLAGS.experiment == 'cifar_dcgan_ae':
        opts = config.cifar_dcgan_ae_opts
    elif FLAGS.experiment == 'cifar_dcgan_patch_moments':
        opts = config.cifar_dcgan_patch_moments_opts
    elif FLAGS.experiment == 'celebA_conv_adv':
        opts = config.celebA_conv_adv_opts
    else:
        assert False, "Invalid experiment defaults"

    if FLAGS.dataset is not None:
        opts['dataset'] = FLAGS.dataset
    if FLAGS.z_dim is not None:
        opts['z_dim'] = FLAGS.z_dim
    if FLAGS.lambda_imq is not None:
        opts['lambda_imq'] = FLAGS.lambda_imq
    if FLAGS.experiment_path is not None:
        opts['experiment_path'] = FLAGS.experiment_path
    if FLAGS.encoder_distribution is not None:
        opts['encoder_distribution'] = FLAGS.encoder_distribution
    if FLAGS.z_prior is not None:
        opts['z_prior'] = FLAGS.z_prior
    if FLAGS.loss_reconstruction is not None:
        opts['loss_reconstruction'] = FLAGS.loss_reconstruction
    if FLAGS.disentanglement_metric is not None:
        opts['disentanglement_metric'] = FLAGS.disentanglement_metric
    if FLAGS.make_pictures_every is not None:
        opts['make_pictures_every'] = FLAGS.make_pictures_every
    if FLAGS.save_every is not None:
        opts['save_every'] = FLAGS.save_every
    if FLAGS.batch_size is not None:
        opts['batch_size'] = FLAGS.batch_size
    if FLAGS.encoder_architecture is not None:
        opts['encoder_architecture'] = FLAGS.encoder_architecture
    if FLAGS.decoder_architecture is not None:
        opts['decoder_architecture'] = FLAGS.decoder_architecture
    if FLAGS.z_logvar_regularisation is not None:
        if FLAGS.z_logvar_regularisation == "None" is not None:
            opts['z_logvar_regularisation'] = None
        else:
            opts['z_logvar_regularisation'] = FLAGS.z_logvar_regularisation
    if FLAGS.lambda_logvar_regularisation is not None:
        opts['lambda_logvar_regularisation'] = FLAGS.lambda_logvar_regularisation
    if FLAGS.loss_regulariser is not None:
        if FLAGS.loss_regulariser == "None":
            opts['loss_regulariser'] = None
        else:
            opts['loss_regulariser'] = FLAGS.loss_regulariser
    if FLAGS.beta is not None:
        opts['beta'] = FLAGS.beta
    if FLAGS.plot_losses is not None:
        if FLAGS.plot_losses == "True":
            opts['plot_losses'] = True
        elif FLAGS.plot_losses == "False":
            opts['plot_losses'] = False
    if FLAGS.adversarial_cost_n_filters is not None:
        opts['adversarial_cost_n_filters'] = FLAGS.adversarial_cost_n_filters
    if FLAGS.adv_cost_nlayers is not None:
        opts['adv_cost_nlayers'] = FLAGS.adv_cost_nlayers
    if FLAGS.adversarial_cost_kernel_size is not None:
        opts['adversarial_cost_kernel_size'] = FLAGS.adversarial_cost_kernel_size
    if FLAGS.adv_cost_lambda is not None:
        opts['adv_cost_lambda'] = FLAGS.adv_cost_lambda
    if FLAGS.adv_cost_normalise_filter is not None:
        opts['adv_cost_normalise_filter'] = FLAGS.adv_cost_normalise_filter
    if FLAGS.pixel_wise_l2 is not None:
        opts['pixel_wise_l2'] = FLAGS.pixel_wise_l2
    if FLAGS.encoder_num_filters is not None:
        opts['encoder_num_filters'] = FLAGS.encoder_num_filters
    if FLAGS.decoder_num_filters is not None:
        opts['decoder_num_filters'] = FLAGS.decoder_num_filters
    if FLAGS.encoder_num_layers is not None:
        opts['encoder_num_layers'] = FLAGS.encoder_num_layers
    if FLAGS.decoder_num_layers is not None:
        opts['decoder_num_layers'] = FLAGS.decoder_num_layers
    if FLAGS.l2_lambda is not None:
        opts['l2_lambda'] = FLAGS.l2_lambda
    if FLAGS.patch_classifier_lambda is not None:
        opts['patch_classifier_lambda'] = FLAGS.patch_classifier_lambda

    model = wae.Model(opts)
    model.train()
