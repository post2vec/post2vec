# -*- coding: utf-8 -*-

import torch
from utils.json_util import load_json
import argparse
from utils.time_util import get_current_time
import os


def save_args(args):
    # save args
    import json, os
    from copy import copy

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    arg_fpath = os.path.join(args.save_dir, "args.json")
    arg_dict = vars(copy(args))
    for x in arg_dict:
        arg_dict[x] = str(arg_dict[x])
    with open(arg_fpath, 'w') as f:
        f.write(json.dumps(arg_dict, indent=4))
    print("Saved argument in %s" % arg_fpath)


def load_args(snapshot_dir):
    ############################ model arguments settings ############################
    parser = argparse.ArgumentParser(description='Multi-label Classifier based on Multi-component')
    args = parser.parse_args()
    arg_dict = args.__dict__

    # load arguments from arg.json
    print("Processing snapshot %s" % (snapshot_dir), get_current_time())
    arg_json = load_json(os.path.join(snapshot_dir, "args.json"))
    for key, val in arg_json.items():
        try:
            if key == "device":
                val = -1
            if key != "model_selection":
                val = eval(val)
        except Exception as e:
            pass
        finally:
            arg_dict[key] = val

    arg_dict["train"] = None

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loaded args.", get_current_time())

    return args


def load_model(args, param_fpath):
    # model selection
    if args.model_selection == 'all':
        from main.tag_rec.approaches.post2vec.models.model_all import MultiComp

        model = MultiComp(args)
    elif args.model_selection == 'title':
        from main.tag_rec.approaches.post2vec.models.model_title import MultiComp

        model = MultiComp(args)
    elif args.model_selection == 'title_desc_text':
        from main.tag_rec.approaches.post2vec.models.model_title_desc_text import MultiComp

        model = MultiComp(args)
    else:
        print("No such model!")
        exit()

    print("Inited model %s use param %s." % (args.model_selection, param_fpath), get_current_time())

    model.load_state_dict(torch.load(param_fpath))

    if args.cuda:
        torch.cuda.set_device(-1)
        model = model.cuda()

    print("Loaded model.", get_current_time())

    return model
