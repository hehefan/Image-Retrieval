import cPickle
import sys
import os
from PIL import Image
import numpy as np
import sys

IMG_PATH = "3DLandmark"

def get_data():
  print "Reading data ..."
  sys.stdout.flush()
  train = []
  val = []
  val_flag = []

  for img in open('image.list', 'r'):
    img = img.strip()
    arr = img[:-4].split('_')
    name = arr[0]
    label = int(arr[1]) - 1
    img_path = os.path.join(IMG_PATH, img)
    im = Image.open(img_path)
    im = im.resize((362, 272))
    im = np.array(im)
    im = np.transpose(im, (1, 0, 2))

    if label not in val_flag:
      val_flag.append(label)
      val.append((im, label))
    else:
      train.append((im, label))

  return train, val

