#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
from main.approaches.stoa.data_helpers import batch_iter, load_data_and_labels
from pathConfig import data_dir
from utils.data_util import load_pickle
from utils.vocab_util import vocab_to_index_dict
import sys
from utils.eval_util import evaluate_batch
from utils.time_util import get_current_time
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

app_name = "tagrcnn"
app_dir = os.path.join(sample_K_dir, "approach", app_name)
snapshot_dir = os.path.join(app_dir, "snapshot")
if not os.path.exists(snapshot_dir):
    print("snapshot %s not exist!" % snapshot_dir)
    exit()

# input files
text_vocab_fpath = os.path.join(vocab_dir, "title_desc_text_vocab.pkl")
text_vocab = load_pickle(text_vocab_fpath)
text_vocab = vocab_to_index_dict(vocab=text_vocab, ifpad=True)

tag_vocab_fpath = os.path.join(vocab_dir, "tag_vocab.pkl")
tag_vocab = load_pickle(tag_vocab_fpath)
tag_vocab = vocab_to_index_dict(vocab=tag_vocab, ifpad=False)

# basic path
test_dir = sample_K_dir + os.sep + "test"
print("Setting:\ntask : %s\ndataset : %s\nts : %s\n" % (task, dataset, ts))
snapshot_name = "04-01-19_14-41-13"
checkpoint_dir = os.path.join(app_dir, "snapshot", snapshot_name, "checkpoints")
if not os.path.exists(checkpoint_dir):
    print("check point dir %s not exist!" % checkpoint_dir)
    exit()

#################################################################################


# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("max_len", 400, "Cut max length (default: 600)")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

topk_list = [1, 2, 3, 4, 5]

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        device_count={"GPU": 1},
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_X").outputs[0]
        input_sequence_length = graph.get_operation_by_name("input_sequence_length").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # post2vec
        post2vec = graph.get_tensor_by_name("dropout/post2vec/mul:0")

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # prepare test data
        sample_test_data_dir = os.path.join(sample_K_dir, "sample_test")
        test_data_cnt = 0
        res_str = ''
        for f in os.listdir(sample_test_data_dir):
            print("Processing #%s test data %s..." % (test_data_cnt, f), get_current_time())
            test_data_cnt += 1

            sample_test_data_fpath = os.path.join(sample_test_data_dir, f)
            sample_test_data = load_pickle(sample_test_data_fpath)

            print("# test data = %s" % len(sample_test_data), get_current_time())
            x_test, y_test = load_data_and_labels(qlist=sample_test_data, text_vocab=text_vocab, max_len=FLAGS.max_len,
                                                  tag_vocab=tag_vocab)

            # Generate batches for one epoch
            batches = batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = list()

            batch_cnt = 0

            for x_test_batch in batches:
                if type(x_test_batch) != bool:
                    sequence_length = [len(sample) for sample in x_test_batch]
                    batch_predictions = sess.run(predictions,
                                                 {input_x: x_test_batch, input_sequence_length: sequence_length,
                                                  dropout_keep_prob: 1.0})
                    if len(all_predictions) == 0:
                        print("len(all_predictions) == 0")
                        all_predictions = batch_predictions
                    else:
                        all_predictions = np.concatenate([all_predictions, batch_predictions])
                    batch_cnt += 1
                    print("Batch cnt = %d, shape of batch prediction %s, shape of all prediction %s" % (
                        batch_cnt, batch_predictions.shape, all_predictions.shape))

            # Print accuracy if y_test is defined
            print("all prediction shape %s, y_test shape %s" % (all_predictions.shape, y_test.shape))
            pre, rc, f1, cnt = evaluate_batch(pred=all_predictions, label=y_test, topk_list=topk_list)

            pre[:] = [x / cnt for x in pre]
            rc[:] = [x / cnt for x in rc]
            f1[:] = [x / cnt for x in f1]

            print("# test : %s" % cnt)
            print("Precision\t\t%s" % ("\t".join(str(x) for x in pre)))
            print("Recall\t\t%s" % ("\t".join(str(x) for x in rc)))
            print("F1\t\t%s\n" % ("\t".join(str(x) for x in f1)))
            res_str += ("%s\t\t" % f)
            res_str += ("Precision\t\t%s\t\t" % ("\t".join(str(x) for x in pre)))
            res_str += ("Recall\t\t%s\t\t" % ("\t".join(str(x) for x in rc)))
            res_str += ("F1\t\t%s\n\n" % ("\t".join(str(x) for x in f1)))

res_fpath = os.path.join(app_dir, "%s.res" % snapshot_name)
write_str_to_file(res_str, res_fpath)
