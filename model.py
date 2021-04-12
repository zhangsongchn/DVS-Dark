from __future__ import division
import tensorflow as tf
import copy
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets


def conv2d(inputs, nout, ks, s, padding, name):
  return tf.layers.conv2d(inputs, nout, (ks, ks), (s, s), padding=padding, name=name)

def deconv2d(inputs, nout, ks, s, padding, name):
  return tf.layers.conv2d_transpose(inputs, nout, (ks, ks), (s, s), padding=padding, name=name)

def instance_normalization(inputs, name):
  with tf.variable_scope(name):
    depth = inputs.get_shape()[3]
    scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
    offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
    mean, variance = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (inputs-mean)*inv
    return scale*normalized + offset

def batch_normalization(inputs, istraining, name):
  return tf.layers.batch_normalization(inputs, momentum=0.9, epsilon=1e-5, training=istraining, name=name)


def tv_loss(images):
  pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
  pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
  sum_axis = [1, 2, 3]
  tot_var = (tf.reduce_sum(tf.abs(pixel_dif1), axis=sum_axis) + \
      tf.reduce_sum(tf.abs(pixel_dif2), axis=sum_axis))
  return tot_var

class DarkEnhance(object):
  def __init__(self, args, sess, name="darkenhance"):
    self.input_size = args.input_size
    self.nevent = args.event_channel
    self.ngray = args.gray_channel
    self.nin = self.nevent * 2 + self.ngray
    self.nsample = self.nevent + self.ngray
    self.ngf = args.ngen_filters
    self.ndf = args.ndis_filters
    self.L1_lambda = args.L1_lambda
    self.sess = sess
    self.max_grad_norm = args.max_grad_norm


    with tf.name_scope("data"):
      self.real_images = tf.placeholder(tf.float32, [args.batch_size] + self.input_size + [self.nin], name="real_images")
      self.eventA, self.eventB, self.gray = tf.split(self.real_images, [8, 8, 1], axis=-1)

      self.istraining = tf.placeholder(tf.bool, [], name="istraining")
      self.learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")
    self.batch_size = tf.shape(self.real_images)[0]
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

    with tf.variable_scope("darkenhance"):
      self.b_enc = self.encoder_layer(self.eventB, False, name="generator_encoderE")
      self.a_enc = self.encoder_layer(self.eventA, True, name="generator_encoderE")

      b_noise = self.get_noise(self.eventB)
      self.n_enc = self.encoder_layer(b_noise, False, name="encoderN")
      self.b_enc_res = self.b_enc + self.n_enc

      self.DH_real = self.discriminator(self.a_enc, False, name="discriminatorH")
      self.DH_fake = self.discriminator(self.b_enc_res, True, name="discriminatorH")

      nb_enc = self.encoder_noise(self.b_enc_res, False, name="generator_decoderA")
      self.b_enc_aug = self.b_enc_res + nb_enc

      self.b2g_coarse = self.decoder_layer(self.b_enc_res, False, name="generator_decoderG")
      self.b2g_fine = self.decoder_layer(self.b_enc_aug, True, name="generator_decoderG")
      self.a2g_coarse = self.decoder_layer(self.a_enc, True, name="generator_decoderG")
      na_enc = self.encoder_noise(self.a_enc, True, name="generator_decoderA")
      self.a_enc_aug = self.a_enc + na_enc
      self.a2g_fine = self.decoder_layer(self.a_enc_aug, True, name="generator_decoderG")

      self.n2g = self.decoder_layer(self.n_enc, True, name="generator_decoderG")
      self.b2g_enc = self.decoder_layer(self.b_enc, True, name="generator_decoderG")
      self.d2g = self.decoder_layer(nb_enc, True, name="generator_decoderG")
      self.r2g = self.decoder_layer(self.b_enc_res, True, name="generator_decoderG")


    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)
    self.saver = tf.train.Saver(tf.global_variables())

  def get_noise(self, image):
    b, h, w, _ = image.get_shape().as_list()
    noise = tf.random_normal([b, h, w, 1])
    return tf.concat([image, noise], axis=-1)


  def inference(self, real_images):
    feed_dict = {self.real_images:real_images, self.istraining:False}
    return self.sess.run([self.gray, self.b2g_coarse, self.b2g_fine, self.a2g_coarse, self.a2g_fine], feed_dict=feed_dict)


  def encoder_noise(self, r9, reuse=False, name="encoderN"):
    with tf.variable_scope(name):
      if reuse:
        tf.get_variable_scope().reuse_variables()
      else:
        assert tf.get_variable_scope().reuse is False

      def residule_block(x, dim, ks=3, s=1, name='res'):
        p = int((ks - 1) / 2)
        y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = instance_normalization(conv2d(y, dim, ks, s, padding='valid', name=name+'_c1'), name+'_bn1')
        y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = instance_normalization(conv2d(y, dim, ks, s, padding='valid', name=name+'_c2'), name+'_bn2')
        return y + x

      b, h, w, _ = r9.get_shape().as_list()
      noise = tf.random_normal([b, h, w, 1])
      noise = conv2d(noise, self.ngf, 3, 1, padding='same', name='g_e1_c')
      r9 = tf.concat([r9, noise], axis=-1)
      r9 = tf.nn.relu(instance_normalization(conv2d(r9, self.ngf*4, 3, 1, padding='same', name='g_e2_c'), 'g_e2_bn'))

      r1 = residule_block(r9, self.ngf*4, name='g_r1')
      r2 = residule_block(r1, self.ngf*4, name='g_r2')
      r3 = residule_block(r2, self.ngf*4, name='g_r3')
      r4 = residule_block(r3, self.ngf*4, name='g_r4')
      r5 = residule_block(r4, self.ngf*4, name='g_r5')
      r6 = residule_block(r5, self.ngf*4, name='g_r6')
      r7 = residule_block(r6, self.ngf*4, name='g_r7')
      r8 = residule_block(r7, self.ngf*4, name='g_r8')
      r9 = residule_block(r8, self.ngf*4, name='g_r9')

      return r9

  def encoder_layer(self, image, reuse=False, name="generator_encoder"):
    with tf.variable_scope(name):
      if reuse:
        tf.get_variable_scope().reuse_variables()
      else:
        assert tf.get_variable_scope().reuse is False

      def residule_block(x, dim, ks=3, s=1, name='res'):
        p = int((ks - 1) / 2)
        y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = instance_normalization(conv2d(y, dim, ks, s, padding='valid', name=name+'_c1'), name+'_bn1')
        y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = instance_normalization(conv2d(y, dim, ks, s, padding='valid', name=name+'_c2'), name+'_bn2')
        return y + x

      c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
      c1 = tf.nn.relu(instance_normalization(conv2d(c0, self.ngf, 7, 1, padding='valid', name='g_e1_c'), 'g_e1_bn'))
      c2 = tf.nn.relu(instance_normalization(conv2d(c1, self.ngf*2, 3, 2, padding='same', name='g_e2_c'), 'g_e2_bn'))
      c3 = tf.nn.relu(instance_normalization(conv2d(c2, self.ngf*4, 3, 2, padding='same', name='g_e3_c'), 'g_e3_bn'))

      r1 = residule_block(c3, self.ngf*4, name='g_r1')
      r2 = residule_block(r1, self.ngf*4, name='g_r2')
      r3 = residule_block(r2, self.ngf*4, name='g_r3')
      r4 = residule_block(r3, self.ngf*4, name='g_r4')
      r5 = residule_block(r4, self.ngf*4, name='g_r5')
      r6 = residule_block(r5, self.ngf*4, name='g_r6')
      r7 = residule_block(r6, self.ngf*4, name='g_r7')
      r8 = residule_block(r7, self.ngf*4, name='g_r8')
      r9 = residule_block(r8, self.ngf*4, name='g_r9')

      return r9

  def decoder_layer(self, r9, reuse=False, name="generator_decoder"):
    with tf.variable_scope(name):
      if reuse:
        tf.get_variable_scope().reuse_variables()
      else:
        assert tf.get_variable_scope().reuse is False

      def residule_block(x, dim, ks=3, s=1, name='res'):
        p = int((ks - 1) / 2)
        y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = instance_normalization(conv2d(y, dim, ks, s, padding='valid', name=name+'_c1'), name+'_bn1')
        y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = instance_normalization(conv2d(y, dim, ks, s, padding='valid', name=name+'_c2'), name+'_bn2')
        return y + x

      r1 = residule_block(r9, self.ngf*4, name='g_r1')
      r2 = residule_block(r1, self.ngf*4, name='g_r2')
      r3 = residule_block(r2, self.ngf*4, name='g_r3')
      r4 = residule_block(r3, self.ngf*4, name='g_r4')
      r5 = residule_block(r4, self.ngf*4, name='g_r5')
      r6 = residule_block(r5, self.ngf*4, name='g_r6')
      r7 = residule_block(r6, self.ngf*4, name='g_r7')
      r8 = residule_block(r7, self.ngf*4, name='g_r8')
      r9 = residule_block(r8, self.ngf*4, name='g_r9')

      d1 = deconv2d(r9, self.ngf*2, 3, 2, padding='same', name='g_d1_dc')
      d1 = tf.nn.relu(instance_normalization(d1, 'g_d1_bn'))
      d2 = deconv2d(d1, self.ngf, 3, 2, padding='same', name='g_d2_dc')
      d2 = tf.nn.relu(instance_normalization(d2, 'g_d2_bn'))
      d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
      pred = tf.nn.sigmoid(conv2d(d2, self.ngray, 7, 1, padding='valid', name='g_pred_c'))
      return pred

  def discriminator(self, image, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
      if reuse:
        tf.get_variable_scope().reuse_variables()
      else:
        assert tf.get_variable_scope().reuse is False

      h0 = tf.nn.leaky_relu(conv2d(image, self.ndf, 4, 2, 'same', name='d_h0_conv'))

      h1 = tf.nn.leaky_relu(instance_normalization(conv2d(h0, self.ndf*2, 4, 2, 'same', name='d_h1_conv'), 'd_bn1'))

      h2 = tf.nn.leaky_relu(instance_normalization(conv2d(h1, self.ndf*4, 4, 2, 'same', name='d_h2_conv'), 'd_bn2'))

      h3 = tf.nn.leaky_relu(instance_normalization(conv2d(h2, self.ndf*8, 4, 1, 'same', name='d_h3_conv'), 'd_bn3'))

      h4 = conv2d(h3, 1, 4, 1, 'same', name='d_h3_pred')

      return h4

  def discriminator_srgan(self, image, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
      if reuse:
        tf.get_variable_scope().reuse_variables()
      else:
        assert tf.get_variable_scope().reuse is False

      def discriminator_block(x, dim, ks, s, name):
        y = batch_normalization(conv2d(x, dim, ks, s, padding='valid', name=name+'_c1'), self.istraining, name+'_bn1')
        return tf.nn.leaky_relu(y)

      h0 = tf.nn.leaky_relu(conv2d(image, self.ndf, 3, 1, 'same', name='d_h1_conv'))
      h1 = discriminator_block(h0, self.ndf, 3, 2, name='disblock_1')
      h2 = discriminator_block(h1, self.ndf*2, 3, 1, name='disblock_2')
      h3 = discriminator_block(h2, self.ndf*2, 3, 2, name='disblock_3')
      h4 = discriminator_block(h3, self.ndf*4, 3, 1, name='disblock_4')
      h5 = discriminator_block(h4, self.ndf*4, 3, 2, name='disblock_5')

      h5 = tf.layers.flatten(h5)
      h6 = tf.nn.leaky_relu(tf.layers.dense(h5, 512, name="dense_1"))
      h7 = tf.layers.dense(h6, 1, name="dense_2")
      return h7

  def trainable_vars(self, scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
