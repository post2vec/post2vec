# -*- coding: utf-8 -*-

import numpy as np


def load_npy(fpath):
    return np.load(fpath)


def save_npy(data, fpath):
    np.save(fpath, data)
