from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys
import tensorflow as tf
#from keras.layers.normalization import BatchNormalization
#from keras.layers.convolutional import Conv2D
#from keras.layers.convolutional import MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Conv2D
#from keras.layers.core import Dropout
#from keras.layers.core import Activation
from tensorflow.keras import Model
import load_image as li
import numpy as np

if len(sys.argv) < 2:
    sys.exit()
image_dir = sys.argv[1]
image_dir_2 = sys.argv[2]
image_dir_3 = sys.argv[3]
image_dir_4 = sys.argv[4]

re_shape = (240,240)
yes_data = li.convert_images_to_data(image_dir, re_shape)
no_data = li.convert_images_to_data(image_dir_2, re_shape)
yes_label = np.full((yes_data.shape[0], 1), 1)
no_label = np.full((no_data.shape[0], 1), 0)
yes_test_data = li.convert_images_to_data(image_dir_3, re_shape)
yes_test_label = np.full((yes_test_data.shape[0], 1), 1)
no_test_data = li.convert_images_to_data(image_dir_4, re_shape)
no_test_label = np.full((no_test_data.shape[0], 1), 0)

print(yes_data.shape, no_data.shape)
print(yes_label.shape, no_label.shape)
data = list(yes_data)
data += list(no_data)
data_label = list(yes_label)
data_label += list(no_label)

test_data = list(yes_test_data)
test_data += list(no_test_data)
test_label = list(yes_test_label)
test_label += list(no_test_label)

np_data = np.array(data)
np_label = np.array(data_label)

np_test_data = np.array(test_data)
np_test_label = np.array(test_label)
print(np_data.shape)
print(np_label.shape)
np_data = np_data.astype('float64')
np_label = np_label.astype('float64')
np_test_data = np_test_data.astype('float64')
np_test_label = np_test_label.astype('float64')
(x_train, y_train), (x_test, y_test) = (np_data, np_label), (np_test_data, np_test_label)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
  def __init__(self):
      super(MyModel, self).__init__()
      self.conv1 = Conv2D(3, 3, input_shape=(240, 240), activation='relu')
      self.flatten = Flatten()
      self.d1 = Dense(128, activation='relu')
      self.d2 = Dense(10, activation='softmax')
  def call(self, x):
      x = self.conv1(x)
      x = self.conv1(x)
      x = self.d1(x)
      return self.d2(x)
from keras import backend as K
K.clear_session()
#model = MyModel()
inputShape = (240, 240, 3)
model = tf.keras.models.Sequential([
  #tf.keras.layers.Flatten(input_shape=inputShape),
  #Conv2D(32, (3, 3), padding="same",input_shape=inputShape),
  #tf.keras.layers.Dense(240, activation='relu'),
  #tf.keras.layers.Dropout(0.2),
  #tf.keras.layers.Dense(360, activation='softmax'),
  #tf.keras.layers.Dropout(0.2),
  #tf.keras.layers.Dense(240, activation='relu'),
  #tf.keras.layers.Dropout(0.2),
  #tf.keras.layers.Dense(100, activation='softmax'),
  #tf.keras.layers.Dropout(0.2),
  #tf.keras.layers.Dense(30, activation='relu'),
  #tf.keras.layers.Dropout(0.2),
  #tf.keras.layers.Dense(15, activation='softmax'),
  #tf.keras.layers.Dropout(0.2),
  #tf.keras.layers.Dense(2, activation='relu')
])
chanDim=1
# CONV => RELU => POOL
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
#model.add(tf.keras.layers.Dense(240, activation='softmax'))
model.add(tf.keras.layers.Dense(240, activation='relu'))
#model.add(BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
#model.add(tf.keras.layers.Dropout(0.25))

# (CONV => RELU) * 2 => POOL
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(tf.keras.layers.Dense(64, activation='relu'))
#model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(tf.keras.layers.Dense(64, activation='relu'))
#model.add(BatchNormalization(axis=chanDim))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#model.add(tf.keras.layers.Dropout(0.25))

# (CONV => RELU) * 2 => POOL
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(tf.keras.layers.Dense(32, activation='relu'))
#model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(tf.keras.layers.Dense(32, activation='relu'))
#model.add(BatchNormalization(axis=chanDim))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#model.add(tf.keras.layers.Dropout(0.25))
# first (and only) set of FC => RELU layers
model.add(tf.keras.layers.Flatten())
#model.add(Dense(128))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
#model.add(BatchNormalization())
#model.add(tf.keras.layers.Dropout(0.5))

# use a *softmax* activation for single-label classification
# and *sigmoid* activation for multi-label classification
#model.add(Dense(classes))
model.add(tf.keras.layers.Dense(2, activation='relu'))

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))