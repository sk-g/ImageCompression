from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import train

FLAGS = tf.flags.FLAGS
mock = tf.test.mock


class TrainTest(tf.test.TestCase):

  def _test_build_graph_helper(self, weight_factor):
    FLAGS.max_number_of_steps = 0
    FLAGS.weight_factor = weight_factor

    batch_size = 3
    patch_size = 16

    FLAGS.batch_size = batch_size
    FLAGS.patch_size = patch_size
    mock_imgs = np.zeros([batch_size, patch_size, patch_size, 3],
                         dtype=np.float32)

    with mock.patch.object(train, 'data_provider') as mock_data_provider:
      mock_data_provider.provide_data.return_value = mock_imgs
      train.main(None)

  def test_build_graph_noadversarialloss(self):
    self._test_build_graph_helper(0.0)

  def test_build_graph_adversarialloss(self):
    self._test_build_graph_helper(1.0)


if __name__ == '__main__':
  tf.test.main()