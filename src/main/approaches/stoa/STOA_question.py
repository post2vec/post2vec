# -*- coding: utf-8 -*-

from nltk import word_tokenize


class STOAQuestion:
    # use __slots to decrease the memory usage
    __slots__ = ['qid', 'text', 'creation_date', 'tags']

    def __init__(self, qid, text, creation_date, tags):
        self.qid = qid
        self.text = text
        self.creation_date = creation_date
        self.tags = tags

    def get_comp_by_name(self, comp_name):
        if comp_name == "qid":
            return self.qid
        if comp_name == "text":
            return self.text
        if comp_name == "creation_date":
            return self.creation_date
        if comp_name == "tags":
            return self.tags


# type
Component_group = {"text": ["title", "desc_text"], "code": ["desc_code"]}


def get_title(data):
    return [word_tokenize(d.title.lower()) for d in data]


def get_desc_text(data):
    return [word_tokenize(d.desc_text.lower()) for d in data]


def get_desc_code(data):
    return [word_tokenize(d.desc_code.lower()) for d in data]


def get_tags(data):
    return [d.tags.lower().replace('<', ' ').replace('>', ' ').strip().split() for d in data]
