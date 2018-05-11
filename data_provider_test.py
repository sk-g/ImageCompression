from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

import tensorflow as tf

import data_provider


class DataProviderTest(tf.test.TestCase):

  def _test_data_provider_helper(self, split_name):
    dataset_dir = os.path.join(
        tf.flags.FLAGS.test_srcdir,
        'google3/third_party/tensorflow_models/gan/image_compression/testdata/')

    batch_size = 3
    patch_size = 8
    images = data_provider.provide_data(
        split_name, batch_size, dataset_dir, patch_size=8)
    self.assertListEqual([batch_size, patch_size, patch_size, 3],
                         images.shape.as_list())

    with self.test_session(use_gpu=True) as sess:
      with tf.contrib.slim.queues.QueueRunners(sess):
        images_out = sess.run(images)
        self.assertEqual((batch_size, patch_size, patch_size, 3),
                         images_out.shape)
        # Check range.
        self.assertTrue(np.all(np.abs(images_out) <= 1.0))

  def test_data_provider_train(self):
    self._test_data_provider_helper('train')

  def test_data_provider_validation(self):
    self._test_data_provider_helper('validation')


if __name__ == '__main__':
  tf.test.main()
