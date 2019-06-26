from __future__ import print_function
import sys
import tensorflow as tf
import keras as k

from synthetic_dataset import *


def smooth_L1_loss(y_true, y_pred, total_mask):

    absolute_loss = tf.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred) ** 2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)

    l1_loss = total_mask * l1_loss
    return tf.reduce_sum(l1_loss, axis=-1)


def log_loss(y_true, y_pred):
    '''
    Compute the softmax log loss.
    Arguments:
        y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
            In this context, the expected tensor has shape (batch_size, #boxes, #classes)
            and contains the ground truth bounding box categories.
        y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
            the predicted data, in this context the predicted bounding box categories.
    Returns:
        The softmax log loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
        of shape (batch, n_boxes_total).
    '''
    # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
    y_pred = tf.maximum(y_pred, 1e-15)
    # Compute the log loss
    log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)

    return log_loss


def ssd_loss(y_true, y_pred):

    # tf.Print(tf.shape(y_true), y_true)
    sh = tf.shape(y_true)
    # tf.Print(sh, y_true)
    b = sh[0]
    r = sh[1]
    c = sh[2]
    a = sh[3]
    ch = sh[4]

    box_true = y_true[:, :, :, :, 0:4]
    class_true = y_true[:, :, :, :, 4:]

    box_pred = y_pred[:, :, :, :, 0:4]
    class_pred = y_pred[:, :, :, :, 4:]

    class_true = tf.reshape(class_true, [b * r * c * a, -1])
    class_pred = tf.reshape(class_pred, [b * r * c * a, -1])

    box_true = tf.reshape(box_true, [b * r * c * a, -1])
    box_pred = tf.reshape(box_pred, [b * r * c * a, -1])

    """******************* Getting negative and positive mask as 3:1 ratio********"""
    background_true = y_true[:, :, :, :, -1] > 0
    foreground_true = tf.logical_not(background_true)

    total_background_mask= tf.reshape(background_true, [b * r * c * a])
    total_foreground_mask = tf.reshape(foreground_true, [b * r * c * a])

    # No of positives and negatives
    n_positives = tf.count_nonzero(total_foreground_mask, dtype=tf.int32)
    n_negatives = tf.count_nonzero(total_background_mask, dtype=tf.int32)

    # Ratio is 3:1 negative:positive
    n_required_neg = n_positives * 3

    required_neg_mask = tf.random_normal([b * r * c * a], dtype=tf.float32)
    required_neg_mask = tf.greater_equal(required_neg_mask, tf.to_float(1-n_required_neg/(n_negatives+n_positives)))
    required_neg_mask = tf.logical_and(total_background_mask, required_neg_mask)

    # Final mask of selected boxes including all positives and some negatives
    total_mask = tf.to_float(tf.logical_or(total_foreground_mask, required_neg_mask))

    """*************Calculation of smooth and classification loss*******************"""
    # total_mask_class = tf.reshape(tf.tile(total_mask, [ch-4]), [ -1, ch-4])
    # softmax_pred = tf.nn.softmax(class_pred)
    classification_loss = k.backend.categorical_crossentropy(class_true, class_pred)
    classification_loss = total_mask * classification_loss

    total_mask_box = tf.reshape(tf.tile(total_mask, [4]), [ -1, 4])
    box_loss = smooth_L1_loss(box_true, box_pred, total_mask_box)

    alpha = 1.0
    total_loss = (tf.reduce_sum(classification_loss) + alpha * tf.reduce_sum(box_loss)) / tf.maximum(1.0, tf.to_float(n_positives))  # In case `n_positive == 0`
    loss = total_loss * tf.to_float(b)

    # if 0:
    #     loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
    #     loss = tf.Print(loss, [tf.reduce_sum(box_loss)], message='Loss box loss \t', summarize=1000)
    #     loss = tf.Print(loss, [class_pred], message='class_pred \t', summarize=1000)
    #     loss = tf.Print(loss, [class_true], message='class_true \t', summarize=1000)
    #     loss = tf.Print(loss, [classification_loss], message='classification_loss \t', summarize=1000)
    #     loss = tf.Print(loss, [n_positives], message='n_positives \t', summarize=1000)
    #     loss = tf.Print(loss, [n_negatives], message='n_negatives \t', summarize=1000)
    #     loss = tf.Print(loss, [tf.count_nonzero(required_neg_mask, dtype=tf.int32)], message='Total selected \t', summarize=1000)

    return loss


if __name__ == '__main__':
    syn_gen = SynthenticGenerator(300, 300, batch_size=1)

    x_batch, y_batch = syn_gen.__getitem__(0)
    # syn_gen.get_predict2Boxes(x_batch, y_batch)

    with tf.Session() as sess:
        x = tf.constant(y_batch, tf.float32)
        y = tf.constant(y_batch, tf.float32)
        print(sess.run(ssd_loss(x, y)))
