import os
import sys
sys.path.append('../')
import tensorflow as tf
from vgg import vgg_16
from PIL import Image
import numpy as np
from scipy.spatial import distance
import operator
import commands

slim = tf.contrib.slim
step = 100
tfmodels = '../tfmodel-FT'
IMG_PATH = "data/Oxford5k"
QRY_PATH = "data/Query"


cnt = 0
image_tensor = tf.placeholder(tf.float32, (None, None, None, 3))
sess = tf.Session()
logits, end_points = vgg_16(image_tensor, num_classes=713, is_training=False, spatial_squeeze=False)
saver = tf.train.Saver()

max_mAP_conv5 = 0
max_cnt_conv5 = 0

max_mAP_pool5 = 0
max_cnt_pool5 = 0

max_mAP_fc6 = 0
max_cnt_fc6 = 0

max_mAP_fc7 = 0
max_cnt_fc7 = 0

sess = tf.Session()
while True:
  cnt += step
  if not os.path.isfile(os.path.join(tfmodels, 'ckpt-%d.meta'%cnt)):
    print("%d: %f - %d: %f - %d: %f - %d: %f"%(max_cnt_conv5, max_mAP_conv5, max_cnt_pool5, max_mAP_pool5, max_cnt_fc6, max_mAP_fc6, max_cnt_fc7, max_mAP_fc7))
    break

  ckpt = os.path.join(tfmodels, 'ckpt-%d'%cnt)
  saver.restore(sess, ckpt)

  conv5 = {}
  pool5 = {}
  fc6 = {}
  fc7 = {}

  for img in open('data/image.list', 'r'):
    img = img.strip()
    img_path = os.path.join(IMG_PATH, img)
    im = Image.open(img_path)
    width, height = im.size
    width = max(width, 224)
    height = max(height, 224)
    im = im.resize((width, height))
    im = np.array(im)
    im = np.expand_dims(np.transpose(im, (1, 0, 2)), axis=0)
    f1, f2, f3, f4 = sess.run([end_points['conv5'], end_points['pool5'], end_points['fc6'], end_points['fc7']], feed_dict={image_tensor: im})
    conv5[img[:-4]] = f1[0]
    pool5[img[:-4]] = f2[0]
    fc6[img[:-4]] = f3[0]
    fc7[img[:-4]] = f4[0]

  qry_conv5 = {}
  qry_pool5 = {}
  qry_fc6 = {}
  qry_fc7 = {}
  for img in open('data/query.list', 'r'):
    img = img.strip()
    img_path = os.path.join(QRY_PATH, img)
    im = Image.open(img_path)
    width, height = im.size
    width = max(width, 224)
    height = max(height, 224)
    im = im.resize((width, height))
    im = np.array(im)
    im = np.expand_dims(np.transpose(im, (1, 0, 2)), axis=0)
    f1, f2, f3, f4 = sess.run([end_points['conv5'], end_points['pool5'], end_points['fc6'], end_points['fc7']], feed_dict={image_tensor: im})
    qry_conv5[img[:-4]] = f1[0]
    qry_pool5[img[:-4]] = f2[0]
    qry_fc6[img[:-4]] = f3[0]
    qry_fc7[img[:-4]] = f4[0]

  # conv5
  for query, feature in qry_conv5.iteritems():
    feature /= np.linalg.norm(feature)
    rst = {}
    for img, feat in conv5.iteritems():
      feat /= np.linalg.norm(feat)
      rst[img] = np.linalg.norm(feature - feat)
    rst = sorted(rst.items(), key=operator.itemgetter(1))
    with open('ans/'+query, 'w') as f:
      for (k, v) in rst:
        f.write(k+'\n')

  mAP_conv5 = 0.0
  with open("LIST", "r") as f:
    num = 0.0
    for line in f:
      line = line.strip()
      _, rst = commands.getstatusoutput("./compute_ap data/Groundtruth_files/" + line + " ans/" + line)
      mAP_conv5 += float(rst)
      num += 1
  mAP_conv5 /= num
  if mAP_conv5 > max_mAP_conv5:
    max_mAP_conv5 = mAP_conv5
    max_cnt_conv5 = cnt

  # pool5
  for query, feature in qry_pool5.iteritems():
    feature /= np.linalg.norm(feature)
    rst = {}
    for img, feat in pool5.iteritems():
      feat /= np.linalg.norm(feat)
      rst[img] = np.linalg.norm(feature - feat)
    rst = sorted(rst.items(), key=operator.itemgetter(1))
    with open('ans/'+query, 'w') as f:
      for (k, v) in rst:
        f.write(k+'\n')

  mAP_pool5 = 0.0
  with open("LIST", "r") as f:
    num = 0.0
    for line in f:
      line = line.strip()
      _, rst = commands.getstatusoutput("./compute_ap data/Groundtruth_files/" + line + " ans/" + line)
      mAP_pool5 += float(rst)
      num += 1
  mAP_pool5 /= num
  if mAP_pool5 > max_mAP_pool5:
    max_mAP_pool5 = mAP_pool5
    max_cnt_pool5 = cnt

  # fc6
  for query, feature in qry_fc6.iteritems():
    feature /= np.linalg.norm(feature)
    rst = {}
    for img, feat in fc6.iteritems():
      feat /= np.linalg.norm(feat)
      rst[img] = np.linalg.norm(feature - feat)
    rst = sorted(rst.items(), key=operator.itemgetter(1))
    with open('ans/'+query, 'w') as f:
      for (k, v) in rst:
        f.write(k+'\n')

  mAP_fc6 = 0.0
  with open("LIST", "r") as f:
    num = 0.0
    for line in f:
      line = line.strip()
      _, rst = commands.getstatusoutput("./compute_ap data/Groundtruth_files/" + line + " ans/" + line)
      mAP_fc6 += float(rst)
      num += 1
  mAP_fc6 /= num
  if mAP_fc6 > max_mAP_fc6:
    max_mAP_fc6 = mAP_fc6
    max_cnt_fc6 = cnt

  # fc7
  for query, feature in qry_fc7.iteritems():
    feature /= np.linalg.norm(feature)
    rst = {}
    for img, feat in fc7.iteritems():
      feat /= np.linalg.norm(feat)
      rst[img] = np.linalg.norm(feature - feat)
    rst = sorted(rst.items(), key=operator.itemgetter(1))
    with open('ans/'+query, 'w') as f:
      for (k, v) in rst:
        f.write(k+'\n')

  mAP_fc7 = 0.0
  with open("LIST", "r") as f:
    num = 0.0
    for line in f:
      line = line.strip()
      _, rst = commands.getstatusoutput("./compute_ap data/Groundtruth_files/" + line + " ans/" + line)
      mAP_fc7 += float(rst)
      num += 1
  mAP_fc7 /= num
  if mAP_fc7 > max_mAP_fc7:
    max_mAP_fc7 = mAP_fc7
    max_cnt_fc7 = cnt

  print("STEP: %d - CONV5: %f - POOL5: %f - FC6: %f - FC7: %f"%(cnt, mAP_conv5, mAP_pool5, mAP_fc6, mAP_fc7))
  sys.stdout.flush() 
