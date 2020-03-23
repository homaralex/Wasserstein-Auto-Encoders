import tensorflow as tf
import numpy as np
import os
import models
import utils

import disentanglement_metric


class Model(object):
    def __init__(self, opts, load=False):
        self.sess = tf.Session()

        self.opts = opts
        utils.opts_check(self)

        self.z_dim = self.opts['z_dim']
        self.batch_size = self.opts['batch_size']
        self.train_data, self.test_data = utils.load_data(self, seed=0)

        self.data_dims = self.train_data.shape[1:]
        self.input = tf.placeholder(tf.float32, (None,) + self.data_dims, name="input")

        self.losses_train = []
        self.losses_test_random = []
        self.losses_test_fixed = []

        self.experiment_path = self.opts['experiment_path']

        if load is False:
            utils.create_directories(self)
            utils.save_opts(self)
            utils.copy_all_code(self)

        models.encoder_init(self)
        models.decoder_init(self)
        models.prior_init(self)
        models.loss_init(self)
        models.optimizer_init(self)
        if 'data_augmentation' in self.opts and self.opts['data_augmentation'] is True:
            models.data_augmentation_init(self)

        self.fixed_test_sample = self.sample_minibatch(test=True, seed=0)
        self.fixed_train_sample = self.sample_minibatch(test=False, seed=0)
        self.fixed_codes = self.sample_codes(seed=0)

        if self.opts['make_pictures_every'] is not None:
            utils.plot_all_init(self)

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
        self.sess.run(tf.global_variables_initializer())

        if load is True:
            self.load_saved_model()

    def save(self, it):
        model_path = "checkpoints/model"
        save_path = self.saver.save(self.sess, model_path, global_step=it)
        print("Model saved to: %s" % save_path)

    def restore(self, model_path):
        self.saver.restore(self.sess, model_path)
        print("Model restored from : %s" % model_path)

    def train(self, it=0):
        if 'data_augmentation' in self.opts and self.opts['data_augmentation'] is True:
            augment = True
        else:
            augment = False
        print("Beginning training")
        if self.opts['optimizer'] == 'adam':
            learning_rates = [i[0] for i in self.opts['learning_rate_schedule']]
            iterations_list = [i[1] for i in self.opts['learning_rate_schedule']]
            total_num_iterations = iterations_list[-1]
            lr_counter = 0
            lr = learning_rates[lr_counter]
            lr_iterations = iterations_list[lr_counter]
            while it < total_num_iterations:
                if it % 1000 == 0:
                    print("\nIteration %i" % it, flush=True)
                if it % 100 == 0:
                    print('.', end='', flush=True)
                it += 1
                while it > lr_iterations:
                    lr_counter += 1
                    lr = learning_rates[lr_counter]
                    lr_iterations = iterations_list[lr_counter]

                self.sess.run(
                    self.train_step,
                    feed_dict={self.learning_rate: lr,
                               self.input: self.sample_minibatch(batch_size=self.batch_size, augment=augment)}
                )

                if 'proximal' in self.opts['z_logvar_regularisation']:
                    self.apply_proximal_gradient(lr=lr)

                if self.opts['loss_reconstruction'] in ['L2_squared+adversarial', 'L2_squared+adversarial+l2_filter',
                                                        'L2_squared+multilayer_conv_adv',
                                                        'L2_squared+adversarial+l2_norm', 'normalised_conv_adv']:
                    self.sess.run(
                        self.adv_cost_train_step,
                        feed_dict={self.learning_rate: lr,
                                   self.input: self.sample_minibatch(batch_size=self.batch_size, augment=augment)}
                    )

                if (self.opts['print_log_information'] is True) and (it % 100 == 0):
                    utils.print_log_information(self, it)

                if self.opts['make_pictures_every'] is not None:
                    if it % self.opts['make_pictures_every'] == 0:
                        utils.plot_all(self, it)

                if it % self.opts['save_every'] == 0:
                    self.save(it)

        print('\nComputing test error')
        self.compute_test_error()

        # once training is complete, calculate disentanglement metric
        if 'disentanglement_metric' in self.opts:
            if self.opts['disentanglement_metric'] is True:
                self.disentanglement = disentanglement_metric.Disentanglement(self)
                self.disentanglement.do_all(it)

        # save random samples and test reconstructions for FID scores:
        if 'FID_score_samples' in self.opts:
            if self.opts['FID_score_samples'] is True:
                self.save_FID_samples()

    def compute_test_error(self):
        l = len(self.test_data)
        total_error = 0
        num_batches = l // self.batch_size
        for i in range(num_batches):
            if i % 5 == 0:
                print('.', end='', flush=True)
            batch = self.test_data[self.batch_size * i:self.batch_size * (i + 1)]

            loss_reconstruction = self.sess.run([
                self.loss_reconstruction, ],
                feed_dict={self.input: batch}
            )[0]

            total_error += loss_reconstruction

        total_error /= num_batches
        print(f'\nAverage error: {total_error}')
        with open('test_error.txt', 'w') as out_file:
            out_file.write(str(total_error))

    def encode(self, images, mean=True):
        if mean is False:
            return self.sess.run(self.z_sample, feed_dict={self.input: images})
        if mean is True:
            return self.sess.run(self.z_mean, feed_dict={self.input: images})

    def decode(self, codes):
        return self.sess.run(tf.nn.sigmoid(self.x_logits_img_shape), feed_dict={self.z_sample: codes})

    def sample_codes(self, batch_size=None, seed=None):
        if batch_size is None:
            batch_size = self.batch_size
        if seed is not None:
            st0 = np.random.get_state()
            np.random.seed(seed)
        # z_mean here is just a placeholder so that z_prior_sample knows what size to be
        codes = self.sess.run(self.z_prior_sample, feed_dict={self.z_mean: np.random.randn(batch_size, self.z_dim)})
        if seed is not None:
            np.random.set_state(st0)
        return codes

    def sample_minibatch(self, batch_size=None, test=False, seed=None, augment=False):
        if seed is not None:
            st0 = np.random.get_state()
            np.random.seed(seed)
        if batch_size is None:
            batch_size = self.batch_size
        if test is False:
            sample = self.train_data[np.random.choice(range(len(self.train_data)), batch_size, replace=False)]
        else:
            sample = self.test_data[np.random.choice(range(len(self.test_data)), batch_size, replace=False)]

        if augment is True and 'data_augmentation' in self.opts and self.opts['data_augmentation'] is True:
            sample = self.sess.run(self.distorted_inputs, feed_dict={self.input: sample})
        if seed is not None:
            np.random.set_state(st0)
        return sample

    def load_saved_model(self):
        os.chdir(self.experiment_path)
        self.saver.restore(self.sess, tf.train.latest_checkpoint('checkpoints'))

    def save_FID_samples(self):
        # makes 10,000 random samples and 10,000 train reconstructions
        # (or the whole train set is smaller than 10,000)
        random_samples = []
        test_reconstructions = []
        train_reconstructions = []
        print("Generating random samples: (each . is 5\%)")
        for i in range(100):
            if i % 5 == 0:
                print('.', end='', flush=True)
            codes = self.sample_codes(batch_size=100)
            ims = self.decode(codes)
            random_samples.append(ims)

        random_samples = np.concatenate(random_samples)
        np.save("output/random_samples.npy", random_samples)

        print("Generating test reconstructions:")
        if len(self.test_data) < 10000:
            # reconstruct all data
            l = len(self.test_data)
            for i in range(l // 100):
                if i % 5 == 0:
                    print('.', end='', flush=True)
                batch = self.test_data[100 * i:100 * (i + 1)]
                encoded = self.encode(batch)
                decoded = self.decode(encoded)
                test_reconstructions.append(decoded)
            if 100 * (i + 1) < l:
                batch = self.test_data[100 * (i + 1):]
                encoded = self.encode(batch)
                decoded = self.decode(encoded)
                test_reconstructions.append(decoded)
        else:
            # 10,000 samples
            for i in range(100):
                if i % 5 == 0:
                    print('.', end='', flush=True)
                batch = self.test_data[100 * i:100 * (i + 1)]
                encoded = self.encode(batch)
                decoded = self.decode(encoded)
                test_reconstructions.append(decoded)

        test_reconstructions = np.concatenate(test_reconstructions)
        np.save("output/test_reconstructions.npy", test_reconstructions)

        print("Generating train reconstructions:")
        if len(self.train_data) < 10000:
            # reconstruct all data
            l = len(self.train_data)
            for i in range(l // 100):
                if i % 5 == 0:
                    print('.', end='', flush=True)
                batch = self.train_data[100 * i:100 * (i + 1)]
                encoded = self.encode(batch)
                decoded = self.decode(encoded)
                train_reconstructions.append(decoded)
            if 100 * (i + 1) < l:
                batch = self.train_data[100 * (i + 1):]
                encoded = self.encode(batch)
                decoded = self.decode(encoded)
                train_reconstructions.append(decoded)
        else:
            # 10,000 samples
            for i in range(100):
                if i % 5 == 0:
                    print('.', end='', flush=True)
                batch = self.train_data[100 * i:100 * (i + 1)]
                encoded = self.encode(batch)
                decoded = self.decode(encoded)
                train_reconstructions.append(decoded)

        train_reconstructions = np.concatenate(train_reconstructions)
        np.save("output/train_reconstructions.npy", train_reconstructions)

    def generate_samples(self, num_samples, batch_size=100):
        samples = []

        while len(samples) < num_samples:
            codes = self.sample_codes(batch_size=batch_size)
            ims = self.decode(codes)
            samples.extend(ims)

        return np.array(samples[:num_samples])

    def get_variances(self, num_samples, batch_size=100, test_data=False):
        dset = self.test_data if test_data else self.train_data

        log_vars = []
        i = 0
        while len(log_vars) < num_samples:
            batch = dset[batch_size * i:batch_size * (i + 1)]
            log_var_batch = self.sess.run(self.z_logvar, feed_dict={self.input: batch})
            log_vars.extend(log_var_batch)
            i += 1

        return np.array(log_vars[:num_samples])

    def apply_proximal_gradient(self, lr):
        weight_names = []

        if 'enc' in self.opts['z_logvar_regularisation']:
            weight_names.extend(['z_mean/kernel', 'z_logvar/kernel'])
        if 'dec' in self.opts['z_logvar_regularisation']:
            weight_names.append('dec_first/kernel')
        weights_to_penalize = [w for w in tf.trainable_variables() if any(w_name in w.name for w_name in weight_names)]
        assert len(weights_to_penalize) > 0

        min_val = lr * self.opts['lambda_logvar_regularisation']

        for weight_matrix in weights_to_penalize:
            norms = tf.norm(weight_matrix, ord=2, axis=1 if 'dec' in weight_matrix.name else 0, keepdims=True)

            # TODO is there no other way than to use tf.float.max?
            new_weights = (weight_matrix / norms) * tf.clip_by_value(
                    t=(norms - min_val),
                    clip_value_min=0,
                    clip_value_max=tf.float32.max,
                )
            self.sess.run(weight_matrix.assign(new_weights))

            # self.sess.run(tf.print(tf.math.count_nonzero(weight_matrix)))

    @property
    def enc_batch_norm(self):
        return self.opts.get('enc_batch_norm', True)

    @property
    def dec_batch_norm(self):
        return self.opts.get('dec_batch_norm', True)
