# coding:utf-8
import tensorflow as tf
import numpy as np
from sklearn import metrics
from utils.time_util import get_current_time

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda):
        # Placeholders for input, output and dropout

        # with tf.device('/device:GPU:0'):
        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        # print("sequence_length：",sequence_length)
        # Embedding layer tf.device('/cpu:0') ,

        with tf.name_scope("embedding"):
            print("embedding_textcnn...", get_current_time())
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            print(self.input_x.shape)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer

                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    # self.embedded_chars_x_flat,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                # relu to tanh
                h = tf.nn.tanh(tf.nn.bias_add(conv, b), name="tanh")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        # A dropout layer stochastically “disables” a fraction of its neurons.
        # This prevent neurons from co-adapting and forces them to learn individually useful features.
        # The fraction of neurons we keep enabled is defined by the dropout_keep_prob input to our network.
        # We set this to something like 0.5 during training, and to 1 (disable dropout) during evaluation.

        with tf.name_scope("dropout"):
            print("dropout_textcnn...", get_current_time())
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob, name="post2vec")

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            print("output_textcnn...", get_current_time())
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # sigmoid转化为由0到1的概率
            self.predictions = tf.nn.sigmoid(self.scores, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            print("loss_textcnn...", get_current_time())
            cross_entropy = -tf.reduce_sum(
                (self.input_y * tf.log(self.predictions + 1e-9)) + (1 - self.input_y) * tf.log(
                    1 - self.predictions + 1e-9), name="xentropy")
            # tf.nn.softmax_cross_entropy_with_logits适用于多类别，不适合多标签
            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy + l2_reg_lambda * l2_loss)

        # Accuracy
    # with tf.name_scope("accuracy"):
    # predict_number = tf.cast(tf.reduce_sum(tf.round(self.predictions)),"float")
    # self.predictions = tf.where(tf.cast(tf.round(self.predictions),bool))
    # self.input_y = tf.where(tf.cast(self.input_y,bool))
    # correct_predictions = tf.reduce_sum(tf.equal(tf.where(self.predictions),tf.where(self.input_y)))
    # correct_predictions = correct_predictions/predict_number
    # self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    # correct_predictions = tf.metrics.precision(self.input_y,self.predictions)
    # self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    # def _embedding_lookup_with_zero_padding(params, ids, pad_id=PAD_ID):
    # inputs = tf.nn.embedding_lookup(params, ids)
    # embedding_size = params.get_shape().as_list()[-1]
    # mask = tf.not_equal(ids, pad_id)
    # tile_shape = [1] * mask.get_shape().ndims + [embedding_size]

    # mask = tf.tile(tf.expand_dims(mask, -1), tile_shape)
    # zeros_states = tf.zeros_like(ids, dtype=tf.float32)
    # zeros_states = tf.tile(
    # tf.expand_dims(zeros_states, -1), tile_shape)
    # return tf.where(mask, inputs, zeros_states)
