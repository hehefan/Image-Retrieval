import tensorflow as tf
import numpy as np
import sys
import os

from vgg import vgg_16

slim = tf.contrib.slim

class FineTuneVGG16(object):
  def __init__(self, learning_rate, num_classes=1000, is_training=True):

    self.global_step = tf.Variable(0, trainable=False, name='global_step')

    self.batch_imgs = tf.placeholder(tf.float32, (None, None, None, 3))
    self.batch_lbls = tf.placeholder(tf.int32, (None, 1))

    _, end_points = vgg_16(self.batch_imgs, num_classes, is_training)

    self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.batch_lbls, logits=end_points['vgg_16/fc8'])
    gradients = tf.gradients(self.loss, tf.trainable_variables())
    self.update = tf.train.GradientDescentOptimizer(learning_rate).apply_gradients(zip(gradients, tf.trainable_variables()), global_step=self.global_step)

    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=99999999)

  def step(self, session, batch_imgs, batch_lbls, is_training):
    input_feed = {}
    input_feed[self.batch_imgs] = batch_imgs
    input_feed[self.batch_lbls] = batch_lbls

    if is_training:
      output_feed = [self.loss, self.update]
    else:
      output_feed = [end_points['vgg_16/fc8']]

    outputs = session.run(output_feed, input_feed)
    return outputs[0]

####################################################################################################
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_string("checkpoint_dir", "tfmodel-FT", "Checkpoint directory.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100, "How many training steps to do per checkpoint.")

FLAGS = tf.app.flags.FLAGS

from data import get_data

init_ckpt = './vgg_16.ckpt'

train, validation = get_data()

model = FineTuneVGG16(FLAGS.learning_rate, 713)

exclusions = ['vgg_16/fc8']
variables_to_restore = []
for var in slim.get_model_variables():
  excluded = False
  for exclusion in exclusions:
    if var.op.name.startswith(exclusion):
      excluded = True
      break
  if not excluded:
    variables_to_restore.append(var)

saver = tf.train.Saver(var_list=variables_to_restore)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver.restore(sess, init_ckpt)
  
  current_step = 0
  data = train + validation
  while True:
    np.random.shuffle(data)
    for start,end in zip(range(0, len(data), FLAGS.batch_size), range(FLAGS.batch_size, len(data), FLAGS.batch_size)):
      curr_data = data[start:end]
      imgs = []
      lbls = []
      for (img, lbl) in curr_data:
        imgs.append(img.astype(float))
        lbls.append(lbl)
      step_loss = model.step(sess, np.array(imgs), np.expand_dims(np.array(lbls), axis=1), True)
      current_step += 1
      print ("step %d - loss %.3f" % (current_step, step_loss))
      if current_step % FLAGS.steps_per_checkpoint == 0:
        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'ckpt')
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
      sys.stdout.flush()
