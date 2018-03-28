"""
A collection of models we'll use to attempt to classify videos.
"""
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
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

def unbalanced_loss(y_true, y_pred):
    # y[0] == > scary value
    # y[1] == > non scary value

    # si jamais classe prédite = non peur
    # alors que class vraie = peur
    # c'est une erreur grave

    # si jamais classe prédite = peur
    # alors que classe vraie = non peur
    # c'est une erreur pas grave

    pred_class = K.argmax(y_pred)
    true_class = K.argmax(y_true)

    if (pred_class == true_class):
        return 0

    elif (pred_class == 1 and true_class == 0):
        return 100 * K.mean(K.square(y_pred - y_true), axis=-1)

    elif (pred_class == 0 and true_class == 1):
        return K.mean(K.square(y_pred - y_true), axis=-1)

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

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        #metrics = ['accuracy', tp_self, fp_self, fn_self, tn_self, auc, recall, recall_tf]

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        else:
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()

        # Now compile the network.
        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

        print(self.model.summary())

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(LSTM(2048, return_sequences=True,
                       input_shape=self.input_shape,
                       dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
