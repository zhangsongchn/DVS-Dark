from __future__ import division
from skimage.measure import compare_ssim as ssim
from glob import glob

import tensorflow as tf
import os.path as op
import numpy as np
import random
import time
import math
import cv2
import os
class ImagePool(object):
  def __init__(self, pool_size):
    self.pool_size = pool_size
    self.num_imgs = 0
    self.images = []

  def query(self, images):
    if self.pool_size == 0:
      return image

    return_images = []
    for image in images:
      image = np.expand_dims(image, 0)
      if len(self.images) < self.pool_size:
        self.num_imgs = self.num_imgs + 1
        self.images.append(image)
        return_images.append(image)
      else:
        p = random.uniform(0, 1)
        if p > 0.5:
          random_id = random.randint(0, self.pool_size - 1)
          tmp = self.images[random_id].copy()
          self.images[random_id] = image.copy()
          return_images.append(tmp)
        else:
          return_images.append(image)
    return_images = np.concatenate(return_images, 0)
    return return_images

def imread(image_path):
  return np.expand_dims(cv2.imread(image_path, 0) / 255.0, -1)

def load_train_data(image_path):
  dataA = glob(image_path[0] + "/frame*")[0]
  dataA = np.load(dataA)
  timeA = glob(image_path[0] + "/time*")[0]
  timeA = np.load(timeA)

  dataB = glob(image_path[1] + "/frame*")[0]
  dataB = np.load(dataB)
  timeB = glob(image_path[1] + "/time*")[0]
  timeB = np.load(timeB)
  tmp = np.concatenate([dataA, timeA, dataB, timeB], axis=-1)
  return np.concatenate([timeA, dataA, timeB, dataB], axis=-1)

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image
  return img

def imsave(images, size, path):
  return cv2.imwrite(path, merge(images, size))

def inverse_transform(images):
  return images * 255.0

def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  config_proto = tf.ConfigProto(log_device_placement=log_device_placement,
                                allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto

def get_learning_rate(args, epoch):
  if epoch < args.decay_start:
    return args.learning_rate
  else:
    return args.learning_rate*(args.nb_epoch-epoch)/(args.nb_epoch-args.decay_start)

def get_images(args, model):
  dataA_time = os.listdir(op.join(args.dataset_dir, "testNight"))
  dataA_dirs = []
  for tm in dataA_time:
    ret_dirs = os.listdir(op.join(args.dataset_dir, "testNight", tm))
    ret_dirs = list(map(lambda x: op.join(args.dataset_dir, "testNight", tm, x), ret_dirs))
    dataA_dirs += ret_dirs

  dataB_time = os.listdir(op.join(args.dataset_dir, "testDay"))
  dataB_dirs = []
  for tm in dataB_time:
    ret_dirs = os.listdir(op.join(args.dataset_dir, "testDay", tm))
    ret_dirs = list(map(lambda x: op.join(args.dataset_dir, "testDay", tm, x), ret_dirs))
    dataB_dirs += ret_dirs

  random.shuffle(dataA_dirs)
  random.shuffle(dataB_dirs)
  idx = np.random.randint(min(len(dataA_dirs), len(dataB_dirs))-1-args.batch_size)
  batch_files = list(zip(dataA_dirs[idx:idx+args.batch_size], dataB_dirs[idx:idx+args.batch_size]))
  sample_images = [load_train_data(batch_file) for batch_file in batch_files]
  sample_images = np.array(sample_images).astype(np.float32)
  return model.inference(sample_images)

def load_test_night(image_path):
  dataA = glob(image_path + "/frame*")[0]
  dataA = np.load(dataA)
  dataB = np.zeros([180, 240, 5])
  return np.concatenate([dataA, dataB], axis=-1)

def test_night(args, model):
  dataA_time = os.listdir(op.join(args.dataset_dir, "testNight"))
  for tm in dataA_time:
    if not op.exists(op.join(args.test_save, tm)):
      os.makedirs(op.join(args.test_save, tm))
    dataA_dirs = []
    ret_dirs = os.listdir(op.join(args.dataset_dir, "testNight", tm))
    ret_dirs = list(map(lambda x: op.join(args.dataset_dir, "testNight", tm, x), ret_dirs))
    dataA_dirs += ret_dirs

    dataA_dirs = sorted(dataA_dirs)
    total_batch = len(dataA_dirs) // args.batch_size
    print("test set have %d samples" % len(dataA_dirs))
    for idx in range(total_batch):
      batch_files = list(dataA_dirs[idx * args.batch_size:(idx + 1) * args.batch_size])
      sample_images = [load_test_night(batch_file) for batch_file in batch_files]
      sample_images = np.array(sample_images).astype(np.float32)
      event, fake_B = model.test_night(sample_images)
      event = np.reshape(np.transpose(event, [0, 3, 1, 2]), [args.batch_size, -1, 240, 1])
      saveA2G = np.concatenate([event, fake_B], axis=1)
      save_images(saveA2G, [1, args.batch_size], './{}/{}/img_{:04d}.jpg'.format(args.test_save, tm, idx))


def evaluation(args, model):
  dataA_time = os.listdir(op.join(args.dataset_dir, "testNight"))
  dataA_dirs = []
  for tm in dataA_time:
    ret_dirs = os.listdir(op.join(args.dataset_dir, "testNight", tm))
    ret_dirs = list(map(lambda x: op.join(args.dataset_dir, "testNight", tm, x), ret_dirs))
    dataA_dirs += ret_dirs

  dataB_time = os.listdir(op.join(args.dataset_dir, "testDay"))
  dataB_dirs = []
  for tm in dataB_time:
    ret_dirs = os.listdir(op.join(args.dataset_dir, "testDay", tm))
    ret_dirs = list(map(lambda x: op.join(args.dataset_dir, "testDay", tm, x), ret_dirs))
    dataB_dirs += ret_dirs

  print("test set have %d samples" % min(len(dataA_dirs), len(dataB_dirs)))
  total_batch = min(len(dataA_dirs), len(dataB_dirs)) // args.batch_size
  total_time = 0
  for idx in range(total_batch):
    batch_files = list(zip(dataA_dirs[idx*args.batch_size:(idx+1)*args.batch_size], dataB_dirs[idx*args.batch_size:(idx+1)*args.batch_size]))
    sample_images = [load_train_data(batch_file) for batch_file in batch_files]
    sample_images = np.array(sample_images).astype(np.float32)


    night_path = batch_files[0][0]
    night_path = glob(op.join(night_path, "*.png"))
    night_path = [x for x in night_path if "eventvis" not in x and "shijianvis" not in x]
    night_img = cv2.imread(night_path[0]).astype(np.float)/255
    night_img = night_img[np.newaxis, :,:,0:1,]
    night_frame = sample_images[...,0:1,]
    day_frame = sample_images[...,8:9,]
    
    start_time = time.time()
    gray, Y2G, fY2G, X2G, FX2G = model.inference(sample_images)
    end_time = time.time()
    total_time += (end_time - start_time)

    if not op.exists(args.eval_save+'_FX2G'):
        os.makedirs(args.eval_save+'_FX2G')

    saved = np.concatenate([night_frame, FX2G, night_img], axis=1)
    save_images(saved, [1, args.batch_size], './{}/{}.jpg'.format(args.eval_save+'_FX2G', "".join(batch_files[0][0].split('/')[-2:])))


def test(args, model):
  dataA_time = os.listdir(op.join(args.dataset_dir, "testNight"))
  dataA_dirs = []
  for tm in dataA_time:
    ret_dirs = os.listdir(op.join(args.dataset_dir, "testNight", tm))
    ret_dirs = list(map(lambda x: op.join(args.dataset_dir, "testNight", tm, x), ret_dirs))
    dataA_dirs += ret_dirs

  dataB_time = os.listdir(op.join(args.dataset_dir, "testDay"))
  dataB_dirs = []
  for tm in dataB_time:
    ret_dirs = os.listdir(op.join(args.dataset_dir, "testDay", tm))
    ret_dirs = list(map(lambda x: op.join(args.dataset_dir, "testDay", tm, x), ret_dirs))
    dataB_dirs += ret_dirs

  print("test set have %d samples" % len(dataA_dirs))
  total_batch = min(len(dataA_dirs), len(dataB_dirs)) // args.batch_size

  for idx in range(total_batch):
    batch_files = list(zip(dataA_dirs[idx*args.batch_size:(idx+1)*args.batch_size], dataB_dirs[idx*args.batch_size:(idx+1)*args.batch_size]))
    sample_images = [load_train_data(batch_file) for batch_file in batch_files]
    sample_images = np.array(sample_images).astype(np.float32)

    _, _, gray, fake_G, _, A2G, _ = model.inference(sample_images)

    saveE2G = np.concatenate([fake_G, gray], axis=1)
    saveG2E = np.concatenate([g, fE], axis=1)
    saveA2G = np.concatenate([eA, A2G, rE], axis=1)

    fake_A = fake_A.transpose([3, 0, 1, 2])
    real_A = np.expand_dims(sample_images[0], -1)
    real_B = np.expand_dims(sample_images[-1], -1)
    event_1 = np.expand_dims(sample_images[1], -1)
    event_2 = np.expand_dims(sample_images[2], -1)
    fake_A1 = np.expand_dims(fake_A[0], -1)
    fake_A2 = np.expand_dims(fake_A[1], -1)
    saved = np.concatenate([real_A, event_1, fake_B, fake_A1, fake_A2], axis=1)

    save_images(saved, [args.batch_size, 1], './{}/B_{:04d}.jpg'.format(args.test_save, idx))