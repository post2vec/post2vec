from pathConfig import data_dir
from utils.data_util import random_mini_batch, load_pickle
import os
from utils.padding_and_indexing_util import padding_and_indexing_qlist
from utils.vocab_util import vocab_to_index_dict
from main.approaches.post2vec.post2vec_util import load_args, load_model
from utils.file_util import read_file_str_list, write_str_to_file
from utils.time_util import get_current_time


def get_computed_param(snapshot_fpath):
    str_list = read_file_str_list(snapshot_fpath)
    param_list = list()
    for param in str_list[1:]:
        param_list.append(param.split(',')[0])
    return param_list


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
app_dir = os.path.join(sample_K_dir, "approach", "post2vec")
snapshot_dir = os.path.join(app_dir, "snapshot")

# basic path
print("Setting:\ntask : %s\ndataset : %s\nts : %s\n" % (task, dataset, ts))
#################################################################################

# load vocab
# initial
len_dict_fpath = os.path.join(vocab_dir, "len.pkl")
title_vocab_fpath = os.path.join(vocab_dir, "title_vocab.pkl")
desc_text_vocab_fpath = os.path.join(vocab_dir, "desc_text_vocab.pkl")
desc_code_vocab_fpath = os.path.join(vocab_dir, "desc_code_vocab.pkl")
tag_vocab_fpath = os.path.join(vocab_dir, "tag_vocab.pkl")

# len
# len_dict = load_pickle(len_dict_fpath)
len_dict = dict()
len_dict["max_title_len"] = 100
len_dict["max_desc_text_len"] = 1000
len_dict["max_desc_code_len"] = 1000

# title vocab
title_vocab = load_pickle(title_vocab_fpath)
title_vocab = vocab_to_index_dict(vocab=title_vocab, ifpad=True)

# desc_text vocab
desc_text_vocab = load_pickle(desc_text_vocab_fpath)
desc_text_vocab = vocab_to_index_dict(vocab=desc_text_vocab, ifpad=True)

# desc_code_vocab
desc_code_vocab = load_pickle(desc_code_vocab_fpath)
desc_code_vocab = vocab_to_index_dict(vocab=desc_code_vocab, ifpad=True)

# tag vocab
tag_vocab = load_pickle(tag_vocab_fpath)
tag_vocab = vocab_to_index_dict(vocab=tag_vocab, ifpad=False)

# predict
test_dir = os.path.join(sample_K_dir, "test")

# load args from json file
snapshot_dir_name = "2019-03-29_04-31-59"
param_dir = os.path.join(snapshot_dir, snapshot_dir_name)
args = load_args(param_dir)

topk_list = [1, 2, 3, 4, 5]

res_str = ''
param_name = 'snapshot_steps_649500.pt'
param_fpath = os.path.join(param_dir, param_name)

model = load_model(args, param_fpath)

if args.model_selection == "all":
    from main.tag_rec.approaches.post2vec.models.model_all import eval
elif args.model_selection == "title":
    from main.tag_rec.approaches.post2vec.models.model_title import eval
elif args.model_selection == "title_desc_text":
    from main.tag_rec.approaches.post2vec.models.model_title_desc_text import eval

print("Loading test data...")
# get sample test data
sample_test_data_dir = os.path.join(sample_K_dir, "sample_test")
for f in os.listdir(sample_test_data_dir):
    sample_test_data_fpath = os.path.join(sample_test_data_dir, f)
    sample_test_data = load_pickle(sample_test_data_fpath)
    print("#test data = %s loaded!" % len(sample_test_data), get_current_time())

    processed_test_data = padding_and_indexing_qlist(sample_test_data, len_dict, title_vocab, desc_text_vocab,
                                                     desc_code_vocab, tag_vocab)

    print("random mini batch", get_current_time())
    batches_test = random_mini_batch(processed_test_data, args.batch_size)

    pre, rc, f1, cnt = eval(batches_test, model, args, topk_list)

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

res_fpath = os.path.join(snapshot_dir, "%s-%s-res.csv" % (snapshot_dir_name, param_name))
write_str_to_file(res_str, res_fpath)
