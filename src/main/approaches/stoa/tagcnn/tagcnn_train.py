#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from main.approaches.stoa.data_helpers import load_data_and_labels, batch_iter
from main.approaches.stoa.tagcnn.text_cnn import TextCNN
import sys
from pathConfig import data_dir
from utils.data_util import load_pickle
from utils.vocab_util import vocab_to_index_dict
from utils.file_util import write_str_to_file

################################# data settings #################################
task = 'tagRec'
dataset = "SO-05-Sep-2018"
dataset_dir = data_dir + os.sep + task + os.sep + dataset
# ts dir
ts = 50
ts_dir = dataset_dir + os.sep + "ts%s" % ts
# sample_K dir
sample_K = "test100000"
sample_K_dir = ts_dir + os.sep + "data-%s" % sample_K
vocab_dir = os.path.join(sample_K_dir, "vocab")

app_name = "tagcnn_res"
app_dir = os.path.join(sample_K_dir, "approach", app_name)
if not os.path.exists(app_dir):
    os.mkdir(app_dir)
snapshot_dir = os.path.join(app_dir, "snapshot")
if not os.path.exists(snapshot_dir):
    os.mkdir(snapshot_dir)

# input files
text_vocab_fpath = os.path.join(vocab_dir, "title_desc_text_vocab.pkl")
text_vocab = load_pickle(text_vocab_fpath)
text_vocab = vocab_to_index_dict(vocab=text_vocab, ifpad=True)

tag_vocab_fpath = os.path.join(vocab_dir, "tag_vocab.pkl")
tag_vocab = load_pickle(tag_vocab_fpath)
tag_vocab = vocab_to_index_dict(vocab=tag_vocab, ifpad=False)

# basic path
train_dir = sample_K_dir + os.sep + "train"
print("Setting:\ntask : %s\ndataset : %s\nts : %s\n" % (task, dataset, ts))
#################################################################################

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.7, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.01, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("max_len", 400, "Cut max length (default: 600)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
args_str = ""
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value.value))
    args_str += ("%s = %s\n" % (attr.upper(), value.value))
print("")

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        device_count={"CPU": 10},
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)

    with sess.as_default():

        cnn = TextCNN(
            sequence_length=400,
            num_classes=len(tag_vocab),
            vocab_size=len(text_vocab),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models and summaries
        timestamp = str(time.strftime('%m-%d-%y_%H-%M-%S', time.localtime(time.time())))
        out_dir = os.path.join(snapshot_dir, timestamp)
        print("Writing to {}\n".format(out_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.join(out_dir, "checkpoints")
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        write_str_to_file(args_str, os.path.join(out_dir, "args.json"))
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                # cnn.sequence_length:np.array(x_batch).shape[1],
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss = sess.run(
                [train_op, global_step, cnn.loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                # cnn.sequence_length:x_batch.shape[1],
                cnn.dropout_keep_prob: 1.0
            }
            step, loss = sess.run(
                [global_step, cnn.loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))
            # print (feed_dict)


        # Load data
        print("Loading data...")

        for f in os.listdir(train_dir):
            fpath = os.path.join(train_dir, f)

            train_data = load_pickle(fpath)

            x, y = load_data_and_labels(qlist=train_data, text_vocab=text_vocab, max_len=FLAGS.max_len,
                                        tag_vocab=tag_vocab)

            shuffle_indices = np.random.permutation(np.arange(len(y)))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]

            # Split train/test set
            # TODO: This is very crude, should use cross-validation
            dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
            x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
            y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

            del x, y, x_shuffled, y_shuffled

            print("Vocabulary Size: {:d}".format(len(text_vocab)))
            print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

            # Generate batches
            batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                # if current_step % FLAGS.evaluate_every == 0:
                #     print("\nEvaluation:")
                #     dev_step(x_dev, y_dev, writer=None)
                #     print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
