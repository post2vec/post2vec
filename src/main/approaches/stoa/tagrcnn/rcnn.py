import tensorflow as tf


class RCNN():
    def __init__(self, num_classes, vocab_size,
                 embedding_size, hidden_units,
                 context_size):
        self.X = tf.placeholder(tf.int32, shape=[None, None], name='input_X')
        self.sequence_length = tf.placeholder(tf.int32, shape=[None], name='input_sequence_length')
        self.y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # self.max_sequence_length = tf.placeholder(
        #     tf.int32, shape=None, name='max_sequence_length')
        # if max_sequence_length == 0:
        self.max_sequence_length = tf.reduce_max(self.sequence_length)
        # else:
        # self.max_sequence_length = max_sequence_length
        # self.dropout_keep_prob = tf.placeholder(
        #     tf.float32, name='dropout_keep_prob')

        # l2_loss = tf.constant(0.0)

        with tf.name_scope('embedding'):
            W = tf.Variable(tf.random_uniform(
                [vocab_size, embedding_size], -1.0, 1.0))
            self.embedded_chars = tf.nn.embedding_lookup(W, self.X)

        with tf.name_scope('recurrent'):
            clw1 = tf.Variable(tf.random_normal(
                shape=[1, context_size]), dtype=tf.float32, name='left_context')
            clw1 = tf.tile(clw1, [tf.size(self.sequence_length), 1])
            # Wl = tf.Variable(tf.random_normal(
            #     shape=[context_size, context_size]), dtype=tf.float32)
            # Wsl = tf.Variable(tf.random_normal(
            #     shape=[embedding_size, context_size]), dtype=tf.float32)

            crwn = tf.Variable(tf.random_normal(
                shape=[1, context_size]), dtype=tf.float32, name='right_context')
            crwn = tf.tile(crwn, [tf.size(self.sequence_length), 1])
            # Wr = tf.Variable(tf.random_normal(
            #     shape=[context_size, context_size]), dtype=tf.float32)
            # Wsr = tf.Variable(tf.random_normal(
            #     shape=[embedding_size, context_size]), dtype=tf.float32)

            # print(clw1, crwn)

            lstm_cell = tf.contrib.rnn.BasicRNNCell(
                num_units=context_size, reuse=False)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell, lstm_cell, self.embedded_chars, sequence_length=self.sequence_length, initial_state_fw=clw1,
                initial_state_bw=crwn, dtype=tf.float32)

            left_context = outputs[0][:, :self.max_sequence_length - 1, :]
            clw1 = tf.expand_dims(clw1, 1)
            Cl = tf.concat([clw1, left_context], 1)
            right_context = outputs[1][:, :self.max_sequence_length - 1, :]
            crwn = tf.expand_dims(crwn, 1)
            Cr = tf.concat([crwn, right_context], 1)
            # print(left_context, right_context)
            # clw1 = tf.tile(clw1, [1, max_sequence_length, 1])
            X = tf.concat([Cl, self.embedded_chars, tf.reverse(Cr, [1])], 2)

            W2 = tf.get_variable(
                "W2",
                shape=[embedding_size + 2 * context_size, hidden_units],
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32)
            # W2 = tf.Variable(tf.random_normal(
            # shape=[hidden_units, embedding_size + 2 * context_size],
            # dtype=tf.float32), name='W2')
            b2 = tf.Variable(tf.random_normal(
                shape=[hidden_units], dtype=tf.float32), name='b2')
            self.y2 = tf.tanh(tf.matmul(tf.reshape(
                X, [-1, embedding_size + 2 * context_size]), W2) + b2)
            self.y2 = tf.reshape(
                self.y2, [-1, self.max_sequence_length, hidden_units])

        with tf.name_scope('max_pooling'):
            self.y3 = tf.reduce_max(self.y2, 1, keep_dims=False)

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.y3, self.dropout_keep_prob, name="post2vec")

        with tf.name_scope('output'):
            W4 = tf.get_variable(
                "W4",
                shape=[hidden_units, num_classes],
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32)
            # W4 = tf.Variable(tf.random_normal(
            # shape=[num_classes, hidden_units]), dtype=tf.float32, name='W4')
            b4 = tf.Variable(tf.random_normal(
                shape=[num_classes], dtype=tf.float32), name='b4')
            y4 = tf.matmul(self.h_drop, W4) + b4
            # self.output = tf.reshape(y4)
            self.predictions = tf.nn.sigmoid(y4, name="predictions")

        with tf.name_scope('loss'):
            self.loss_for_1 = - \
                tf.reduce_mean(tf.reduce_sum(
                    self.y * tf.log(self.predictions + 1e-9), 1))
            self.loss_for_0 = - \
                tf.reduce_mean(tf.reduce_sum(
                    (1 - self.y) * tf.log(1 - self.predictions + 1e-9), 1))
            # losses = tf.nn.sigmoid_cross_entropy_with_logits(
            #     logits=y4, labels=self.y)
            # self.loss = tf.reduce_mean(tf.reduce_sum(losses, 1))
            self.loss = self.loss_for_1 + self.loss_for_0

        with tf.name_scope('accuracy'):
            mask = tf.equal(tf.constant(1.), self.y)
            labels_size = tf.count_nonzero(self.y, dtype=tf.int32)
            top10 = tf.nn.top_k(self.predictions, 10, sorted=False)
            kth = tf.reduce_min(top10.values)
            predictions = tf.cast(tf.greater_equal(
                self.predictions, kth), tf.float32)
            hit_10 = tf.count_nonzero(tf.boolean_mask(
                predictions, mask), dtype=tf.int32)
            # hit_10 = tf.size(tf.sets.set_intersection(indices, indices_10))
            # print(hit_10, tf.size(self.sequence_length))
            self.precise_10 = hit_10 / \
                              (tf.size(self.sequence_length) * 10)
            self.recall_10 = hit_10 / labels_size
            top5 = tf.nn.top_k(self.predictions, 5, sorted=False)
            kth = tf.reduce_min(top5.values)
            predictions = tf.cast(tf.greater_equal(
                self.predictions, kth), tf.float32)
            hit_5 = tf.count_nonzero(tf.boolean_mask(
                predictions, mask), dtype=tf.int32)
            # hit_5 = tf.size(tf.sets.set_intersection(indices, indices_5))
            self.precise_5 = hit_5 / \
                             (tf.size(self.sequence_length) * 5)
            self.recall_5 = hit_5 / labels_size
