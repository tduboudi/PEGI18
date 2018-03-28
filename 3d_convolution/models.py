"""
A collection of models we'll use to attempt to classify videos.
"""
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.convolutional import (MaxPooling3D, Conv3D)
from collections import deque
from keras import backend as K

from parameters import param

import tensorflow as tf
import sys

def tp(y_true, y_pred):
   score, up_opt = tf.metrics.true_positives(y_true, y_pred)
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score

def tn(y_true, y_pred):
   score, up_opt = tf.metrics.true_negatives(y_true, y_pred)
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score

def fp(y_true, y_pred):
   score, up_opt = tf.metrics.false_positives(y_true, y_pred)
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score

def fn(y_true, y_pred):
   score, up_opt = tf.metrics.false_negatives(y_true, y_pred)
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score

def recall_tf(y_true, y_pred):
   score, up_opt = tf.metrics.recall(y_true, y_pred)
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score

def auc(y_true, y_pred):
   score, up_opt = tf.metrics.auc(y_true, y_pred)
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score

def tp_self(y_true, y_pred):
    y_pred = K.round(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    return true_positives

def tn_self(y_true, y_pred):
    y_pred = K.round(y_pred)
    true_positives = K.sum(K.round(K.clip((K.ones_like(y_true) - y_true) * (K.ones_like(y_pred) - y_pred), 0, 1)))
    return true_positives

def fn_self(y_true, y_pred):
    y_pred = K.round(y_pred)
    true_positives = K.sum(K.round(K.clip((K.ones_like(y_true) - y_true) * y_pred, 0, 1)))
    return true_positives

def fp_self(y_true, y_pred):
    y_pred = K.round(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * (K.ones_like(y_pred) - y_pred), 0, 1)))
    return true_positives

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 0.01)
    return recall

class ResearchModels():
    def __init__(self, nb_classes, seq_length, saved_model=None, features_length=2048):
        """
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        metrics = ['accuracy', auc, recall, recall_tf]

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        else:
            print("Loading Conv3D")
            self.input_shape = (seq_length, *param['image_shape'])
            self.model = self.conv_3d()

        # Now compile the network.
        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

        print(self.model.summary())

    def conv_3d(self):
        """
        Build a 3D convolutional network, based loosely on C3D.
            https://arxiv.org/pdf/1412.0767.pdf
        """
        # Model.
        model = Sequential()
        model.add(Conv3D( 32, (3,3,3), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(64, (3,3,3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(128, (3,3,3), activation='relu'))
        model.add(Conv3D(128, (3,3,3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(256, (2,2,2), activation='relu'))
        model.add(Conv3D(256, (2,2,2), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Dropout(0.5))
        model.add(Dense(1024))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
