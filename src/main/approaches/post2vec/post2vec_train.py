import torch
from pathConfig import data_dir, project_dir
import argparse
from utils.data_util import random_mini_batch, load_pickle
import os
import datetime
from utils.time_util import get_current_time
from main.approaches.post2vec.post2vec_util import save_args
from utils.padding_and_indexing_util import padding_and_indexing_qlist
from utils.vocab_util import vocab_to_index_dict

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

app_name = "post2vec"
app_dir = os.path.join(sample_K_dir, "approach", app_name)

# input files
# len_dict_fpath = os.path.join(vocab_dir, "len.pkl")
title_vocab_fpath = os.path.join(vocab_dir, "title_vocab.pkl")
desc_text_vocab_fpath = os.path.join(vocab_dir, "desc_text_vocab.pkl")
desc_code_vocab_fpath = os.path.join(vocab_dir, "desc_code_vocab.pkl")
tag_vocab_fpath = os.path.join(vocab_dir, "tag_vocab.pkl")

# basic path
train_dir = sample_K_dir + os.sep + "train"
test_dir = sample_K_dir + os.sep + "test"
print("Setting:\ntask : %s\ndataset : %s\nts : %s\n" % (task, dataset, ts))
#################################################################################


############################ model arguments settings ############################
parser = argparse.ArgumentParser(description='Multi-label Classifier based on Multi-component')
# basic settings
parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=16, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=128, help='batch size for training [default: 64]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=200,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500,
                    help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1,
                    help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-train', type=str, default=True, help='train the model')

############################# tuned parameter #############################
parser.add_argument('-model-selection', type=str, default='title_desc_text',
                    help='model selection [default: all]')  # 'all', 'title', 'title_desc_text'
parser.add_argument('-embed-dim', type=int, default=32,
                    help='number of embedding dimension [default: 128]')  # 32, 64, 128
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')  # 100
parser.add_argument('-title-kernel-sizes', type=str, default='1,2,3',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('-desc-text-kernel-sizes', type=str, default='1,2,3',  # 1,2,3 or 2,3,4 or 3,4,5
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('-desc-code-kernel-sizes', type=str, default='2,3,4',  # 1,2,3 or 2,3,4 or 3,4,5
                    help='comma-separated kernel size to use for convolution')
############################################################################

args = parser.parse_args()
# initial
# len
# len_dict = load_pickle(len_dict_fpath)
len_dict = dict()
len_dict["max_title_len"] = 100
len_dict["max_desc_text_len"] = 1000
len_dict["max_desc_code_len"] = 1000
args.max_title_len = len_dict["max_title_len"]
args.max_desc_text_len = len_dict["max_desc_text_len"]
args.max_desc_code_len = len_dict["max_desc_code_len"]
# title vocab
title_vocab = load_pickle(title_vocab_fpath)
title_vocab = vocab_to_index_dict(vocab=title_vocab, ifpad=True)
args.title_embed_num = len(title_vocab)

# desc_text vocab
desc_text_vocab = load_pickle(desc_text_vocab_fpath)
desc_text_vocab = vocab_to_index_dict(vocab=desc_text_vocab, ifpad=True)
args.desc_text_embed_num = len(desc_text_vocab)

# desc_code_vocab
desc_code_vocab = load_pickle(desc_code_vocab_fpath)
desc_code_vocab = vocab_to_index_dict(vocab=desc_code_vocab, ifpad=True)
args.desc_code_embed_num = len(desc_code_vocab)

# tag vocab
tag_vocab = load_pickle(tag_vocab_fpath)
tag_vocab = vocab_to_index_dict(vocab=tag_vocab, ifpad=False)
args.class_num = len(tag_vocab)

args.title_kernel_sizes = [int(k) for k in args.title_kernel_sizes.split(',')]
args.desc_text_kernel_sizes = [int(k) for k in args.desc_text_kernel_sizes.split(',')]
args.desc_code_kernel_sizes = [int(k) for k in args.desc_code_kernel_sizes.split(',')]

# Device configuration
args.cuda = (not args.no_cuda) and torch.cuda.is_available()
del args.no_cuda

snap_shot_dir = os.path.join(app_dir, "snapshot")
args.save_dir = os.path.join(snap_shot_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

# model selection
if args.model_selection == 'all':
    from main.tag_rec.approaches.post2vec.models.model_all import MultiComp, train

    model = MultiComp(args=args)
elif args.model_selection == 'title':
    from main.tag_rec.approaches.post2vec.models.model_title import MultiComp, train

    model = MultiComp(args=args)

elif args.model_selection == 'title_desc_text':
    from main.tag_rec.approaches.post2vec.models.model_title_desc_text import MultiComp, train

    model = MultiComp(args=args)

else:
    print("No such model!")
    exit()

if args.snapshot is not None:
    print('\nLoading parameter from {}...'.format(args.snapshot))
    model.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    model = model.cuda()

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_args(args)
#################################################################################


try:
    global_train_step = 0
    train_cnt = 0
    for f in sorted(os.listdir(train_dir)):
        print("\n\n# train file = %s" % train_cnt, get_current_time())
        train_cnt += 1
        train_data_fpath = os.path.join(train_dir, f)
        train_data = load_pickle(train_data_fpath)
        print("padding and indexing train", get_current_time())
        processed_train_data = padding_and_indexing_qlist(train_data, len_dict, title_vocab, desc_text_vocab,
                                                          desc_code_vocab, tag_vocab)
        print("random mini batch train", get_current_time())
        batches_train = random_mini_batch(processed_train_data, args.batch_size)
        print("Start train %s..." % f, get_current_time())
        model, global_train_step = train(train_iter=batches_train, dev_iter=None, model=model, args=args,
                                         global_train_step=global_train_step)
except KeyboardInterrupt:
    print('\n' + '-' * 89)
print('Exiting from training early')
