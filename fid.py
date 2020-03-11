'''
From https://github.com/tsc2017/Frechet-Inception-Distance
Code derived from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py

Usage:
    Call get_fid(images1, images2)
Args:
    images1, images2: Numpy arrays with values ranging from 0 to 255 and shape in the form [N, 3, HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be arbitrary. 
    dtype of the images is recommended to be np.uint8 to save CPU memory.
Returns:
    Frechet Inception Distance between the two image distributions.
'''
import argparse
import pickle
from pathlib import Path

import tensorflow as tf
import functools
import numpy as np
import time
from tensorflow.python.ops import array_ops
from tqdm import tqdm

import wae


def inception_activations(images, num_splits=1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.compat.v1.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits=num_splits)
    activations = tf.map_fn(
        fn=functools.partial(tfgan.eval.run_inception, output_tensor='pool_3:0'),
        elems=array_ops.stack(generated_images_list),
        parallel_iterations=8,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')
    activations = array_ops.concat(array_ops.unstack(activations), 0)
    return activations


def get_inception_activations(inps, batch_size):
    n_batches = int(np.ceil(float(inps.shape[0]) / batch_size))
    act = np.zeros([inps.shape[0], 2048], dtype=np.float32)
    for i in tqdm(range(n_batches)):
        inp = inps[i * batch_size: (i + 1) * batch_size] / 255. * 2 - 1
        act[i * batch_size: i * batch_size + min(batch_size, inp.shape[0])] = session.run(activations, feed_dict={
            inception_images: inp})
    return act


def activations2distance(act1, act2):
    return session.run(fcd, feed_dict={activations1: act1, activations2: act2})


def preprocess_images(images):
    return images.transpose(0, 3, 1, 2) * 255.


def get_fid(images1, images2, preprocess, batch_size):
    if preprocess:
        images1, images2 = preprocess_images(images1), preprocess_images(images2)

    assert (type(images1) == np.ndarray)
    assert (len(images1.shape) == 4)
    assert (images1.shape[1] == 3)
    assert (np.min(images1[0]) >= 0 and np.max(images1[0]) > 10), 'Image values should be in the range [0, 255]'
    assert (type(images2) == np.ndarray)
    assert (len(images2.shape) == 4)
    assert (images2.shape[1] == 3)
    assert (np.min(images2[0]) >= 0 and np.max(images2[0]) > 10), 'Image values should be in the range [0, 255]'
    assert (images1.shape == images2.shape), 'The two numpy arrays must have the same shape'
    print('Calculating FID with %i images from each distribution' % (images1.shape[0]))
    start_time = time.time()
    act1 = get_inception_activations(images1, batch_size=batch_size)
    act2 = get_inception_activations(images2, batch_size=batch_size)
    fid = activations2distance(act1, act2)
    print('FID calculation time: %f s' % (time.time() - start_time))
    return fid


def load_model(experiment_path, dataset=None):
    with open(experiment_path + "/opts.pickle", 'rb') as f:
        opts = pickle.load(f)
    if dataset is not None:
        opts['dataset'] = dataset
    # TODO write this in a nicer way
    opts['experiment_path'] = 'results_download/Wasserstein-Auto-Encoders/' + opts['experiment_path']
    model = wae.Model(opts, load=True)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-n', '--num_samples', type=int, default=10000)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-e', '--experiment_path', type=str, required=True)
    parser.add_argument('-d', '--dataset', type=str, default=None)
    args = parser.parse_args()

    tfgan = tf.contrib.gan

    session = tf.compat.v1.InteractiveSession()

    # Run images through Inception.
    inception_images = tf.compat.v1.placeholder(tf.float32, [None, 3, None, None])
    activations1 = tf.compat.v1.placeholder(tf.float32, [None, None], name='activations1')
    activations2 = tf.compat.v1.placeholder(tf.float32, [None, None], name='activations2')
    fcd = tfgan.eval.frechet_classifier_distance_from_activations(activations1, activations2)

    activations = inception_activations(images=inception_images)

    model = load_model(experiment_path=args.experiment_path, dataset=args.dataset)
    real_images = model.test_data[:args.num_samples]
    generated_images = model.generate_samples(num_samples=args.num_samples, batch_size=args.batch_size)

    fid = get_fid(real_images, generated_images, preprocess=True, batch_size=args.batch_size)

    # load_model changes cwd to experiment_path (in wae.Model init) so we can save the results here simply
    with open('test_fid.txt', 'w') as out_file:
        out_file.write(str(fid))
    print(fid)
