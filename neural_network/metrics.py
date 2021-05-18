#==============================================================================#
#  Original Work Author:   Dominik Müller                                      #
#  Copyright:              2020 IT-Infrastructure for                          #
#                          Translational Medical Research,                     #
#                          University of Augsburg                              #
#                                                                              #
#  Derivative Work Author: Jim Heo                                             #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf
import numpy as np


#-----------------------------------------------------#
#                 Precision & Recall                  #
#-----------------------------------------------------#
def precision(y_true, y_pred):
    y_true_yn = K.round(K.clip(y_true, 0, 1))
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))

    # True Positive
    tp = K.sum(y_true_yn * y_pred_yn) 
    # True Positive + False Positive
    tp_fp = K.sum(y_pred_yn)

    precision = tp / (tp_fp + K.epsilon())

    return precision

def recall(y_true, y_pred):
    y_true_yn = K.round(K.clip(y_true, 0, 1))
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))

    # True Positive
    tp = K.sum(y_true_yn * y_pred_yn) 
    # True Positive + False Negative
    tp_fn = K.sum(y_true_yn)
    
    recall = tp / (tp_fn + K.epsilon())

    return recall

#-----------------------------------------------------#
#                    F1-Score loss                    #
#-----------------------------------------------------#
def f1_score(y_true, y_pred):
    _recall = recall(y_true, y_pred)
    _precision = precision(y_true, y_pred)
    _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())
    
    return _f1score

def f1_loss(y_true, y_pred):
    return 1 - f1_score(y_true, y_pred)

#-----------------------------------------------------#
#            Categorical Crossentropy loss            #
#-----------------------------------------------------#
def categorical_crossentropy(y_true, y_pred):
    # Obtain Crossentropy
    crossentropy = K.categorical_crossentropy(y_true, y_pred)
    
    return K.mean(crossentropy)

#-----------------------------------------------------#
#               Binary Crossentropy loss              #
#-----------------------------------------------------#
def binary_crossentropy(y_true, y_pred):
    # Obtain Crossentropy
    return K.sum(K.binary_crossentropy(y_true, y_pred))

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed

def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    if isinstance(alpha, list):
        alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed


def binary_f1_focal(y_truth, y_pred):
    # Obtain Dice Loss without Background
    f1 = f1_loss(y_truth, y_pred)
    # Obtain Focal Loss
    gamma = 2.
    alpha = .25
    focal_loss = binary_focal_loss(gamma, alpha)
    focal = focal_loss(y_truth, y_pred)
    # Return sum
    return f1 + focal

def categorical_f1_focal(y_truth, y_pred):
    # Obtain Dice Loss without Background
    f1 = f1_loss(y_truth, y_pred)
    # Obtain Focal Loss
    gamma = 2.
    alpha = [[.5, .25, .25]]
    focal_loss = categorical_focal_loss(gamma, alpha)
    focal = focal_loss(y_truth, y_pred)
    # Return sum
    return f1 + focal