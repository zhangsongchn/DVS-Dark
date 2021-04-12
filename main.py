from __future__ import print_function, division
from model import DarkEnhance
import tensorflow as tf
import os.path as op
import numpy as np
import random
import config
import utils
import time
import os

def imgs_save(args, ret_list, epoch, idx):
  gray, b2g_coarse, b2g_fine, a2g_coarse, a2g_fine = ret_list
  saved = np.concatenate([b2g_coarse, b2g_fine, gray, a2g_coarse, a2g_fine], axis=1)
  utils.save_images(saved, [1, args.batch_size], './{}/image_{:03d}_{:04d}.jpg'.format(args.img_dir, epoch, idx))

def evaluation(args, model):
  tf.reset_default_graph()
  utils.evaluation(args, model)


def test(args):
  if not op.exists(args.test_save):
    os.makedirs(args.test_save)
  latest_ckpt = tf.train.latest_checkpoint(args.save_dir)
  config_proto = utils.get_config_proto()
  sess = tf.Session(config=config_proto)
  model = DarkEnhance(args, sess, name="darkenhance")

  model.saver.restore(sess, latest_ckpt)
  print('start testing ...')
  start_time = time.time()
  evaluation(args,model)
  end_time = time.time()

if __name__ == '__main__':
  args = config.get_args()
  test(args)