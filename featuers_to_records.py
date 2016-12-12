#!/usr/bin/env python
"""
/***************************************************************************
 

                              -------------------
        begin                : 2016-11-12
        git sha              : $Format:%H$
        copyright            : (C) 2016 by Nejc Dougan
        email                : nejc.dougan@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import numpy as np
import os, glob
import scipy.misc
import tensorflow as tf

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  images = list(data_set[:,0])
  labels = list(data_set[:,1])
  num_examples = len(images)

  if len(labels) != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (len(labels), num_examples))
  rows = np.shape(images)[1]
  cols = np.shape(images)[2]
  depth = np.shape(images)[3]
  num_of_labels = len(labels)

  filename = os.path.join(path , name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    labels_raw = labels[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'num_of_labels': _int64_feature(num_of_labels),
        'label': _bytes_feature(labels_raw),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
	path = '/media/nejc/Prostor/Dropbox/dev/Data/'
	name = '01'
	features = np.load(path + name + '.npy')
	convert_to(features, name)
