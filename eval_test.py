from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import eval


class EvalTest(tf.test.TestCase):

  def test_build_graph(self):
    eval.main(None, run_eval_loop=False)


if __name__ == '__main__':
  tf.test.main()