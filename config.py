fading_squares_opts = {}
fading_squares_opts['dataset'] = 'fading_squares'
fading_squares_opts['experiment_path'] = 'experiments/fading_squares/exp1'
fading_squares_opts['z_dim'] = 2
fading_squares_opts['print_log_information'] = True
fading_squares_opts['make_pictures_every'] = 10000
fading_squares_opts['save_every'] = 10000
fading_squares_opts['plot_axis_walks'] = True
fading_squares_opts['axis_walk_range'] = 1
fading_squares_opts['plot_losses'] =  True
fading_squares_opts['print_log_information'] = True
fading_squares_opts['batch_size'] = 100
fading_squares_opts["encoder_architecture"] = 'FC_dsprites'
fading_squares_opts["decoder_architecture"] = 'FC_dsprites'
fading_squares_opts['z_mean_activation'] = None
fading_squares_opts['encoder_distribution'] = 'uniform'
fading_squares_opts['logvar-clipping'] = [-20,5]
fading_squares_opts['z_prior'] = 'uniform'
fading_squares_opts['loss_reconstruction'] = 'bernoulli'
fading_squares_opts['loss_regulariser'] = 'WAE_MMD'
fading_squares_opts['lambda_imq'] = 20.0
coeffs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0]
fading_squares_opts['IMQ_length_params'] = [c*fading_squares_opts['z_dim'] for c in coeffs]
fading_squares_opts['z_logvar_regularisation'] = None
#fading_squares_opts['lambda_logvar_regularisation'] = 0.5
fading_squares_opts['optimizer'] = 'adam'
fading_squares_opts['learning_rate_schedule'] = [(1e-3, 10000), (7e-4, 20000), (3e-4, 30000), (1e-4, 40001), ]


dsprites_opts = {}
dsprites_opts['dataset'] = 'dsprites'
dsprites_opts['experiment_path'] = 'experiments/dsprites/exp1'
dsprites_opts['z_dim'] = 16
dsprites_opts['print_log_information'] = True
dsprites_opts['make_pictures_every'] = 10000
dsprites_opts['save_every'] = 10000
dsprites_opts['plot_axis_walks'] = False
#dsprites_opts['axis_walk_range'] = 1
dsprites_opts['plot_losses'] =  False
dsprites_opts['print_log_information'] = True
dsprites_opts['batch_size'] = 100
dsprites_opts["encoder_architecture"] = 'FC_dsprites'
dsprites_opts["decoder_architecture"] = 'FC_dsprites'
dsprites_opts['z_mean_activation'] = 'tanh'
dsprites_opts['encoder_distribution'] = 'gaussian'
dsprites_opts['logvar-clipping'] = [-20,5]
dsprites_opts['z_prior'] = 'gaussian'
dsprites_opts['loss_reconstruction'] = 'bernoulli'
dsprites_opts['loss_regulariser'] = 'WAE_MMD'
dsprites_opts['lambda_imq'] = 20.0
coeffs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0]
dsprites_opts['IMQ_length_params'] = [c*dsprites_opts['z_dim'] for c in coeffs]
dsprites_opts['z_logvar_regularisation'] = "L1"
dsprites_opts['lambda_logvar_regularisation'] = 3.0
dsprites_opts['optimizer'] = 'adam'
#dsprites_opts['learning_rate_schedule'] = [(1e-3, 10)]
dsprites_opts['learning_rate_schedule'] = [(1e-3, 15000), (7e-4, 30000), (3e-4, 45000), (1e-4, 60001)]
dsprites_opts['disentanglement_metric'] = True

celebA_mini_opts = {}
celebA_mini_opts['dataset'] = 'celebA_mini'
celebA_mini_opts['experiment_path'] = 'experiments/celebA_mini/exp1'
celebA_mini_opts['z_dim'] = 256
celebA_mini_opts['print_log_information'] = True
celebA_mini_opts['make_pictures_every'] = 10000
celebA_mini_opts['save_every'] = 10000
celebA_mini_opts['plot_axis_walks'] = False
#celebA_mini_opts['axis_walk_range'] = 1
celebA_mini_opts['plot_losses'] =  False
celebA_mini_opts['print_log_information'] = True
celebA_mini_opts['batch_size'] = 100
celebA_mini_opts["encoder_architecture"] = 'small_convolutional_celebA'
celebA_mini_opts["decoder_architecture"] = 'small_convolutional_celebA'
celebA_mini_opts['z_mean_activation'] = None
celebA_mini_opts['encoder_distribution'] = 'gaussian'
celebA_mini_opts['logvar-clipping'] = [-20,5]
celebA_mini_opts['z_prior'] = 'gaussian'
celebA_mini_opts['loss_reconstruction'] = 'bernoulli'
celebA_mini_opts['loss_regulariser'] = 'WAE_MMD'
celebA_mini_opts['lambda_imq'] = 400.0
coeffs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0]
celebA_mini_opts['IMQ_length_params'] = [c*celebA_mini_opts['z_dim'] for c in coeffs]
celebA_mini_opts['z_logvar_regularisation'] = "L1"
celebA_mini_opts['lambda_logvar_regularisation'] = 0.1
celebA_mini_opts['optimizer'] = 'adam'
celebA_mini_opts['learning_rate_schedule'] = [(1e-4, 40000), (1e-5, 80001)]


celebA_random_col_dec_opts = {}
celebA_random_col_dec_opts['dataset'] = 'celebA'
celebA_random_col_dec_opts['experiment_path'] = 'experiments/celebA_col_dec/exp1'
celebA_random_col_dec_opts['z_dim'] = 256
celebA_random_col_dec_opts['print_log_information'] = True
celebA_random_col_dec_opts['make_pictures_every'] = 10000
celebA_random_col_dec_opts['save_every'] = 10000
celebA_random_col_dec_opts['plot_axis_walks'] = False
#celebA_random_col_dec_opts['axis_walk_range'] = 1
celebA_random_col_dec_opts['plot_losses'] =  False
celebA_random_col_dec_opts['print_log_information'] = True
celebA_random_col_dec_opts['batch_size'] = 100
celebA_random_col_dec_opts["encoder_architecture"] = 'small_convolutional_celebA'
celebA_random_col_dec_opts["decoder_architecture"] = 'small_convolutional_celebA'
celebA_random_col_dec_opts['z_mean_activation'] = None
celebA_random_col_dec_opts['encoder_distribution'] = 'gaussian'
celebA_random_col_dec_opts['logvar-clipping'] = [-20,5]
celebA_random_col_dec_opts['z_prior'] = 'gaussian'
celebA_random_col_dec_opts['loss_reconstruction'] = 'bernoulli'
celebA_random_col_dec_opts['loss_regulariser'] = 'WAE_MMD'
celebA_random_col_dec_opts['lambda_imq'] = 400.0
coeffs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0]
celebA_random_col_dec_opts['IMQ_length_params'] = [c*celebA_random_col_dec_opts['z_dim'] for c in coeffs]
celebA_random_col_dec_opts['z_logvar_regularisation'] = 'col_L1_dec'
celebA_random_col_dec_opts['lambda_logvar_regularisation'] = 0.1 / 50
celebA_random_col_dec_opts['optimizer'] = 'adam'
celebA_random_col_dec_opts['learning_rate_schedule'] = [(1e-4, 40000), (1e-5, 80001)]


celebA_random_opts = {}
celebA_random_opts['dataset'] = 'celebA'
celebA_random_opts['experiment_path'] = 'experiments/celebA/exp1'
celebA_random_opts['z_dim'] = 256
celebA_random_opts['print_log_information'] = True
celebA_random_opts['make_pictures_every'] = 10000
celebA_random_opts['save_every'] = 10000
celebA_random_opts['plot_axis_walks'] = False
#celebA_random_opts['axis_walk_range'] = 1
celebA_random_opts['plot_losses'] =  False
celebA_random_opts['print_log_information'] = True
celebA_random_opts['batch_size'] = 100
celebA_random_opts["encoder_architecture"] = 'small_convolutional_celebA'
celebA_random_opts["decoder_architecture"] = 'small_convolutional_celebA'
celebA_random_opts['z_mean_activation'] = None
celebA_random_opts['encoder_distribution'] = 'gaussian'
celebA_random_opts['logvar-clipping'] = [-20,5]
celebA_random_opts['z_prior'] = 'gaussian'
celebA_random_opts['loss_reconstruction'] = 'bernoulli'
celebA_random_opts['loss_regulariser'] = 'WAE_MMD'
celebA_random_opts['lambda_imq'] = 400.0
coeffs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0]
celebA_random_opts['IMQ_length_params'] = [c*celebA_random_opts['z_dim'] for c in coeffs]
celebA_random_opts['z_logvar_regularisation'] = "L1"
celebA_random_opts['lambda_logvar_regularisation'] = 0.1
celebA_random_opts['optimizer'] = 'adam'
celebA_random_opts['learning_rate_schedule'] = [(1e-4, 40000), (1e-5, 80001)]



celebA_deterministic_opts = {}
celebA_deterministic_opts['dataset'] = 'celebA'
celebA_deterministic_opts['experiment_path'] = 'experiments/celebA/exp1'
celebA_deterministic_opts['z_dim'] = 64
celebA_deterministic_opts['print_log_information'] = True
celebA_deterministic_opts['make_pictures_every'] = 10000
celebA_deterministic_opts['save_every'] = 10000
celebA_deterministic_opts['plot_axis_walks'] = False
#celebA_deterministic_opts['axis_walk_range'] = 1
celebA_deterministic_opts['plot_losses'] =  True
celebA_deterministic_opts['print_log_information'] = True
celebA_deterministic_opts['batch_size'] = 100
celebA_deterministic_opts["encoder_architecture"] = 'small_convolutional_celebA'
celebA_deterministic_opts["decoder_architecture"] = 'small_convolutional_celebA'
celebA_deterministic_opts['z_mean_activation'] = None
celebA_deterministic_opts['encoder_distribution'] = 'deterministic'
celebA_deterministic_opts['logvar-clipping'] = None
celebA_deterministic_opts['z_prior'] = 'gaussian'
celebA_deterministic_opts['loss_reconstruction'] = 'bernoulli'
celebA_deterministic_opts['loss_regulariser'] = 'WAE_MMD'
celebA_deterministic_opts['lambda_imq'] = 400.0
coeffs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0]
celebA_deterministic_opts['IMQ_length_params'] = [c*celebA_deterministic_opts['z_dim'] for c in coeffs]
celebA_deterministic_opts['z_logvar_regularisation'] = None
celebA_deterministic_opts['optimizer'] = 'adam'
celebA_deterministic_opts['learning_rate_schedule'] = [(1e-4, 40000), (1e-5, 80001)]


celebA_dcgan_deterministic_opts = {}
celebA_dcgan_deterministic_opts['dataset'] = 'celebA'
celebA_dcgan_deterministic_opts['experiment_path'] = 'experiments/celebA/exp1'
celebA_dcgan_deterministic_opts['z_dim'] = 64
celebA_dcgan_deterministic_opts['print_log_information'] = True
celebA_dcgan_deterministic_opts['make_pictures_every'] = 10000
celebA_dcgan_deterministic_opts['save_every'] = 10000
celebA_dcgan_deterministic_opts['plot_axis_walks'] = False
#celebA_dcgan_deterministic_opts['axis_walk_range'] = 1
celebA_dcgan_deterministic_opts['plot_losses'] =  True
celebA_dcgan_deterministic_opts['print_log_information'] = True
celebA_dcgan_deterministic_opts['batch_size'] = 100
celebA_dcgan_deterministic_opts["encoder_architecture"] = 'dcgan'
celebA_dcgan_deterministic_opts["decoder_architecture"] = 'dcgan'

celebA_dcgan_deterministic_opts['encoder_num_filters'] = 1024
celebA_dcgan_deterministic_opts['encoder_num_layers'] = 4
celebA_dcgan_deterministic_opts['decoder_num_filters'] = 1024
celebA_dcgan_deterministic_opts['decoder_num_layers'] = 4
celebA_dcgan_deterministic_opts['conv_filter_dim'] = 4

celebA_dcgan_deterministic_opts['z_mean_activation'] = None
celebA_dcgan_deterministic_opts['encoder_distribution'] = 'deterministic'
celebA_dcgan_deterministic_opts['logvar-clipping'] = None
celebA_dcgan_deterministic_opts['z_prior'] = 'gaussian'
celebA_dcgan_deterministic_opts['loss_reconstruction'] = 'bernoulli'
celebA_dcgan_deterministic_opts['loss_regulariser'] = 'WAE_MMD'
celebA_dcgan_deterministic_opts['lambda_imq'] = 400.0
coeffs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0]
celebA_dcgan_deterministic_opts['IMQ_length_params'] = [c*celebA_dcgan_deterministic_opts['z_dim'] for c in coeffs]
celebA_dcgan_deterministic_opts['z_logvar_regularisation'] = None
celebA_dcgan_deterministic_opts['optimizer'] = 'adam'
celebA_dcgan_deterministic_opts['learning_rate_schedule'] = [(1e-4, 40000), (1e-5, 80001)]
celebA_dcgan_deterministic_opts['FID_score_samples'] = True



grassli_WAE_opts = {}
grassli_WAE_opts['dataset'] = 'grassli'
grassli_WAE_opts['experiment_path'] = 'experiments/grassli/WAE/exp1'
grassli_WAE_opts['z_dim'] = 64
grassli_WAE_opts['print_log_information'] = True
grassli_WAE_opts['make_pictures_every'] = 10000
grassli_WAE_opts['save_every'] = 10000
grassli_WAE_opts['plot_axis_walks'] = False
#grassli_WAE_opts['axis_walk_range'] = 1
grassli_WAE_opts['plot_losses'] =  True
grassli_WAE_opts['print_log_information'] = True
grassli_WAE_opts['batch_size'] = 100
grassli_WAE_opts["encoder_architecture"] = 'dcgan'
grassli_WAE_opts["decoder_architecture"] = 'dcgan'

grassli_WAE_opts['encoder_num_filters'] = 1024
grassli_WAE_opts['encoder_num_layers'] = 4
grassli_WAE_opts['decoder_num_filters'] = 1024
grassli_WAE_opts['decoder_num_layers'] = 4
grassli_WAE_opts['conv_filter_dim'] = 4

grassli_WAE_opts['z_mean_activation'] = None
grassli_WAE_opts['encoder_distribution'] = 'deterministic'
grassli_WAE_opts['logvar-clipping'] = None
grassli_WAE_opts['z_prior'] = 'gaussian'
grassli_WAE_opts['loss_reconstruction'] = 'L2_squared'
grassli_WAE_opts['loss_regulariser'] = 'WAE_MMD'
grassli_WAE_opts['lambda_imq'] = 400.0
coeffs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0]
grassli_WAE_opts['IMQ_length_params'] = [c*grassli_WAE_opts['z_dim'] for c in coeffs]
grassli_WAE_opts['z_logvar_regularisation'] = None
grassli_WAE_opts['optimizer'] = 'adam'
grassli_WAE_opts['learning_rate_schedule'] = [(1e-4, 40000), (1e-5, 80001)]


grassli_VAE_opts = {}
grassli_VAE_opts['dataset'] = 'grassli'
grassli_VAE_opts['experiment_path'] = 'experiments/grassli/VAE/exp1'
grassli_VAE_opts['z_dim'] = 64
grassli_VAE_opts['print_log_information'] = True
grassli_VAE_opts['make_pictures_every'] = 10000
grassli_VAE_opts['save_every'] = 10000
grassli_VAE_opts['plot_axis_walks'] = False
grassli_VAE_opts['plot_losses'] =  True
grassli_VAE_opts['print_log_information'] = True
grassli_VAE_opts['batch_size'] = 100
grassli_VAE_opts["encoder_architecture"] = 'dcgan'
grassli_VAE_opts["decoder_architecture"] = 'dcgan'
grassli_VAE_opts['encoder_num_filters'] = 1024
grassli_VAE_opts['encoder_num_layers'] = 4
grassli_VAE_opts['decoder_num_filters'] = 1024
grassli_VAE_opts['decoder_num_layers'] = 4
grassli_VAE_opts['conv_filter_dim'] = 4
grassli_VAE_opts['z_mean_activation'] = None
grassli_VAE_opts['encoder_distribution'] = 'gaussian'
grassli_VAE_opts['logvar-clipping'] = None
grassli_VAE_opts['z_prior'] = 'gaussian'
grassli_VAE_opts['loss_reconstruction'] = 'L2_squared'
grassli_VAE_opts['loss_regulariser'] = 'VAE'
grassli_VAE_opts['z_logvar_regularisation'] = None
grassli_VAE_opts['optimizer'] = 'adam'
grassli_VAE_opts['learning_rate_schedule'] = [(1e-4, 40000), (1e-5, 80001)]


celebA_dcgan_adv_cost_opts = {}
celebA_dcgan_adv_cost_opts['dataset'] = 'celebA'
celebA_dcgan_adv_cost_opts['experiment_path'] = 'experiments/celebA/dcgan/adv_cost/exp1'
celebA_dcgan_adv_cost_opts['z_dim'] = 64
celebA_dcgan_adv_cost_opts['print_log_information'] = True
celebA_dcgan_adv_cost_opts['make_pictures_every'] = 1000
celebA_dcgan_adv_cost_opts['save_every'] = 10000
celebA_dcgan_adv_cost_opts['plot_axis_walks'] = False
#celebA_dcgan_adv_cost_opts['axis_walk_range'] = 1
celebA_dcgan_adv_cost_opts['plot_losses'] =  True
celebA_dcgan_adv_cost_opts['print_log_information'] = True
celebA_dcgan_adv_cost_opts['batch_size'] = 100
celebA_dcgan_adv_cost_opts["encoder_architecture"] = 'dcgan'
celebA_dcgan_adv_cost_opts["decoder_architecture"] = 'dcgan'

celebA_dcgan_adv_cost_opts['encoder_num_filters'] = 1024
celebA_dcgan_adv_cost_opts['encoder_num_layers'] = 4
celebA_dcgan_adv_cost_opts['decoder_num_filters'] = 1024
celebA_dcgan_adv_cost_opts['decoder_num_layers'] = 4
celebA_dcgan_adv_cost_opts['conv_filter_dim'] = 4

celebA_dcgan_adv_cost_opts['z_mean_activation'] = None
celebA_dcgan_adv_cost_opts['encoder_distribution'] = 'deterministic'
celebA_dcgan_adv_cost_opts['logvar-clipping'] = None
celebA_dcgan_adv_cost_opts['z_prior'] = 'gaussian'
celebA_dcgan_adv_cost_opts['loss_reconstruction'] = 'L2_squared+adversarial'
celebA_dcgan_adv_cost_opts['adversarial_cost_n_filters'] = 4
celebA_dcgan_adv_cost_opts['adversarial_cost_kernel_size'] = 3
celebA_dcgan_adv_cost_opts['loss_regulariser'] = 'WAE_MMD'
celebA_dcgan_adv_cost_opts['lambda_imq'] = 400.0
coeffs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0]
celebA_dcgan_adv_cost_opts['IMQ_length_params'] = [c*celebA_dcgan_adv_cost_opts['z_dim'] for c in coeffs]
celebA_dcgan_adv_cost_opts['z_logvar_regularisation'] = None
celebA_dcgan_adv_cost_opts['optimizer'] = 'adam'
celebA_dcgan_adv_cost_opts['learning_rate_schedule'] = [(1e-4, 40000), (1e-5, 80001)]
celebA_dcgan_adv_cost_opts['adv_cost_learning_rate_schedule'] = [(1e-5, 40000), (1e-6, 80001)]
celebA_dcgan_adv_cost_opts['FID_score_samples'] = True




celebA_dcgan_adv_cost_l2_filters_opts = {}
celebA_dcgan_adv_cost_l2_filters_opts['dataset'] = 'celebA'
celebA_dcgan_adv_cost_l2_filters_opts['experiment_path'] = 'experiments/celebA/dcgan/adv_cost/exp1'
celebA_dcgan_adv_cost_l2_filters_opts['z_dim'] = 64
celebA_dcgan_adv_cost_l2_filters_opts['print_log_information'] = True
celebA_dcgan_adv_cost_l2_filters_opts['make_pictures_every'] = 1000
celebA_dcgan_adv_cost_l2_filters_opts['save_every'] = 10000
celebA_dcgan_adv_cost_l2_filters_opts['plot_axis_walks'] = False
#celebA_dcgan_adv_cost_l2_filters_opts['axis_walk_range'] = 1
celebA_dcgan_adv_cost_l2_filters_opts['plot_losses'] =  True
celebA_dcgan_adv_cost_l2_filters_opts['print_log_information'] = True
celebA_dcgan_adv_cost_l2_filters_opts['batch_size'] = 100
celebA_dcgan_adv_cost_l2_filters_opts["encoder_architecture"] = 'dcgan'
celebA_dcgan_adv_cost_l2_filters_opts["decoder_architecture"] = 'dcgan'

celebA_dcgan_adv_cost_l2_filters_opts['encoder_num_filters'] = 1024
celebA_dcgan_adv_cost_l2_filters_opts['encoder_num_layers'] = 4
celebA_dcgan_adv_cost_l2_filters_opts['decoder_num_filters'] = 1024
celebA_dcgan_adv_cost_l2_filters_opts['decoder_num_layers'] = 4
celebA_dcgan_adv_cost_l2_filters_opts['conv_filter_dim'] = 4

celebA_dcgan_adv_cost_l2_filters_opts['z_mean_activation'] = None
celebA_dcgan_adv_cost_l2_filters_opts['encoder_distribution'] = 'deterministic'
celebA_dcgan_adv_cost_l2_filters_opts['logvar-clipping'] = None
celebA_dcgan_adv_cost_l2_filters_opts['z_prior'] = 'gaussian'
celebA_dcgan_adv_cost_l2_filters_opts['loss_reconstruction'] = 'L2_squared+adversarial+l2_filter'
celebA_dcgan_adv_cost_l2_filters_opts['adv_cost_lambda'] = 1.0
celebA_dcgan_adv_cost_l2_filters_opts['adversarial_cost_n_filters'] = 4
celebA_dcgan_adv_cost_l2_filters_opts['adversarial_cost_kernel_size'] = 3
celebA_dcgan_adv_cost_l2_filters_opts['loss_regulariser'] = 'WAE_MMD'
celebA_dcgan_adv_cost_l2_filters_opts['lambda_imq'] = 400.0
coeffs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0]
celebA_dcgan_adv_cost_l2_filters_opts['IMQ_length_params'] = [c*celebA_dcgan_adv_cost_l2_filters_opts['z_dim'] for c in coeffs]
celebA_dcgan_adv_cost_l2_filters_opts['z_logvar_regularisation'] = None
celebA_dcgan_adv_cost_l2_filters_opts['optimizer'] = 'adam'
celebA_dcgan_adv_cost_l2_filters_opts['learning_rate_schedule'] = [(1e-4, 40000), (1e-5, 80001)]
celebA_dcgan_adv_cost_l2_filters_opts['adv_cost_learning_rate_schedule'] = [(1e-5, 40000), (1e-6, 80001)]
celebA_dcgan_adv_cost_l2_filters_opts['FID_score_samples'] = True




cifar_dcgan_ae_opts = {}
cifar_dcgan_ae_opts['dataset'] = 'cifar'
cifar_dcgan_ae_opts['experiment_path'] = 'experiments/cifar/dcgan/unregularised/exp1'
cifar_dcgan_ae_opts['z_dim'] = 64
cifar_dcgan_ae_opts['print_log_information'] = True
cifar_dcgan_ae_opts['make_pictures_every'] = 1000
cifar_dcgan_ae_opts['save_every'] = 10000
cifar_dcgan_ae_opts['plot_axis_walks'] = False
#cifar_dcgan_ae_opts['axis_walk_range'] = 1
cifar_dcgan_ae_opts['plot_losses'] =  True
cifar_dcgan_ae_opts['print_log_information'] = True
cifar_dcgan_ae_opts['batch_size'] = 100
cifar_dcgan_ae_opts["encoder_architecture"] = 'dcgan'
cifar_dcgan_ae_opts["decoder_architecture"] = 'dcgan'

cifar_dcgan_ae_opts['encoder_num_filters'] = 1024
cifar_dcgan_ae_opts['encoder_num_layers'] = 4
cifar_dcgan_ae_opts['decoder_num_filters'] = 1024
cifar_dcgan_ae_opts['decoder_num_layers'] = 4
cifar_dcgan_ae_opts['conv_filter_dim'] = 4
cifar_dcgan_ae_opts['data_augmentation'] = True

cifar_dcgan_ae_opts['z_mean_activation'] = None
cifar_dcgan_ae_opts['encoder_distribution'] = 'deterministic'
cifar_dcgan_ae_opts['logvar-clipping'] = None
cifar_dcgan_ae_opts['z_prior'] = 'gaussian'
cifar_dcgan_ae_opts['loss_reconstruction'] = 'L2_squared'
cifar_dcgan_ae_opts['adv_cost_lambda'] = 1.0
cifar_dcgan_ae_opts['adversarial_cost_n_filters'] = 4
cifar_dcgan_ae_opts['adversarial_cost_kernel_size'] = 3
cifar_dcgan_ae_opts['loss_regulariser'] = None
cifar_dcgan_ae_opts['lambda_imq'] = 400.0
coeffs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0]
cifar_dcgan_ae_opts['IMQ_length_params'] = [c*cifar_dcgan_ae_opts['z_dim'] for c in coeffs]
cifar_dcgan_ae_opts['z_logvar_regularisation'] = None
cifar_dcgan_ae_opts['optimizer'] = 'adam'
cifar_dcgan_ae_opts['learning_rate_schedule'] = [(1e-4, 40000), (1e-5, 80001)]
cifar_dcgan_ae_opts['adv_cost_learning_rate_schedule'] = [(1e-5, 40000), (1e-6, 80001)]
cifar_dcgan_ae_opts['FID_score_samples'] = True



celebA_conv_adv_opts = {}
celebA_conv_adv_opts['dataset'] = 'celebA'
celebA_conv_adv_opts['experiment_path'] = 'experiments/celebA/dcgan/unregularised/exp1'
celebA_conv_adv_opts['z_dim'] = 64
celebA_conv_adv_opts['print_log_information'] = True
celebA_conv_adv_opts['make_pictures_every'] = 5000
celebA_conv_adv_opts['save_every'] = 5000
celebA_conv_adv_opts['plot_axis_walks'] = False
#celebA_conv_adv_opts['axis_walk_range'] = 1
celebA_conv_adv_opts['plot_losses'] =  True
celebA_conv_adv_opts['print_log_information'] = True
celebA_conv_adv_opts['batch_size'] = 100
celebA_conv_adv_opts["encoder_architecture"] = 'dcgan'
celebA_conv_adv_opts["decoder_architecture"] = 'dcgan'

celebA_conv_adv_opts['encoder_num_filters'] = 1024
celebA_conv_adv_opts['encoder_num_layers'] = 4
celebA_conv_adv_opts['decoder_num_filters'] = 1024
celebA_conv_adv_opts['decoder_num_layers'] = 4
celebA_conv_adv_opts['conv_filter_dim'] = 4

celebA_conv_adv_opts['z_mean_activation'] = None
celebA_conv_adv_opts['encoder_distribution'] = 'deterministic'
celebA_conv_adv_opts['logvar-clipping'] = None
celebA_conv_adv_opts['z_prior'] = 'gaussian'

celebA_conv_adv_opts['loss_reconstruction'] = 'normalised_conv_adv'
celebA_conv_adv_opts['adv_cost_lambda'] = 1.0
celebA_conv_adv_opts['patch_classifier_lambda'] = 1.0
celebA_conv_adv_opts['l2_lambda'] = 1.0
celebA_conv_adv_opts['adversarial_cost_n_filters'] = 128
celebA_conv_adv_opts['adv_use_sq_features'] = True

celebA_conv_adv_opts['loss_regulariser'] = None
celebA_conv_adv_opts['lambda_imq'] = 400.0
coeffs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0]
celebA_conv_adv_opts['IMQ_length_params'] = [c*celebA_conv_adv_opts['z_dim'] for c in coeffs]
celebA_conv_adv_opts['z_logvar_regularisation'] = None
celebA_conv_adv_opts['optimizer'] = 'adam'
celebA_conv_adv_opts['learning_rate_schedule'] = [(1e-4, 40000), (1e-5, 80001)]
celebA_conv_adv_opts['adv_cost_learning_rate_schedule'] = [(1e-5, 40000), (1e-6, 80001)]
celebA_conv_adv_opts['FID_score_samples'] = True
