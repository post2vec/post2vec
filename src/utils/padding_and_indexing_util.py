import numpy as np


def indexing_label(label, label_vocab):
    embedd_zero = [0 for i in range(len(label_vocab))]
    for l in label:
        if l in label_vocab:
            embedd_zero[label_vocab[l]] = 1
    return np.array(embedd_zero)


def padding_and_indexing_sent(sent, length, vocab):
    new = []
    for w in sent:
        if w in vocab:
            new.append(vocab[w])
            if len(new) == length:
                return np.array(new)
    add = length - len(new)
    for a in range(0, add):
        new.append(vocab['<PAD>'])
    return np.array(new)


def padding_and_indexing_q(q, len_dict, title_vocab, desc_text_vocab, desc_code_vocab, tag_vocab):
    # pre processing, padding and indexing
    q.title = padding_and_indexing_sent(sent=q.title, length=len_dict["max_title_len"], vocab=title_vocab)
    q.desc_text = padding_and_indexing_sent(sent=q.desc_text, length=len_dict["max_desc_text_len"],
                                            vocab=desc_text_vocab)
    q.desc_code = padding_and_indexing_sent(sent=q.desc_code, length=len_dict["max_desc_code_len"],
                                            vocab=desc_code_vocab)
    q.tags = indexing_label(label=q.tags, label_vocab=tag_vocab)
    if sum(q.tags) != 0:
        return q
    return None


def padding_and_indexing_qlist(qlist, len_dict, title_vocab, desc_text_vocab, desc_code_vocab, tag_vocab):
    new_qlist = list()
    for q in qlist:
        # pre processing, padding and indexing
        q = padding_and_indexing_q(q, len_dict, title_vocab, desc_text_vocab, desc_code_vocab, tag_vocab)
        if q:
            new_qlist.append(q)

    return new_qlist


def padding_and_indexing_qlist_without_tag(qlist, len_dict, title_vocab, desc_text_vocab, desc_code_vocab):
    """
    without tag to reduce the size
    :param qlist:
    :param len_dict:
    :param title_vocab:
    :param desc_text_vocab:
    :param desc_code_vocab:
    :return:
    """
    print()
    new_qlist = list()
    for q in qlist:
        # pre processing, padding and indexing
        q.title = padding_and_indexing_sent(sent=q.title, length=len_dict["max_title_len"], vocab=title_vocab)
        q.desc_text = padding_and_indexing_sent(sent=q.desc_text, length=len_dict["max_desc_text_len"],
                                                vocab=desc_text_vocab)
        q.desc_code = padding_and_indexing_sent(sent=q.desc_code, length=len_dict["max_desc_code_len"],
                                                vocab=desc_code_vocab)
        if q:
            new_qlist.append(q)

    return new_qlist
