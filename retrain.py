# Python script to retrain Google Inception model 

# Imports 
import os
import re
import tensorflow as tf
#import tensorflow.python.platform
#from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
import argparse
import sys
import tarfile
from tensorflow.python.platform import gfile
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
from six.moves import urllib
#matplotlib inline
import pickle


FLAGS=None
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def create_inception_graph():
  """"Creates a graph from saved GraphDef file and returns a Graph object.
  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
  with tf.Session() as sess:
    model_filename = os.path.join(
        FLAGS.model_dir, 'classify_image_graph_def.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
#      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
#          tf.import_graph_def(graph_def, name='', return_elements=[
#              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
#              RESIZED_INPUT_TENSOR_NAME]))
#  return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def retrain():
  """
  Retrain inception model
  """
  create_inception_graph()

def main(_):
  #maybe_download_and_extract()
  retrain()
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
      create_inception_graph())


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--train_data',
      type=str,
      default='',
      help='Path to the train data'
  )
  parser.add_argument(
      '--test_data',
      type=str,
      default='',
      help='Path to test data.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
