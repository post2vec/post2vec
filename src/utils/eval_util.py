# -*- coding: utf-8 -*-


def evaluate(pred, label, topk):
    """
    dimension of pred and label should be equal.
    :param pred: a list of prediction
    :param label: a list of true label
    :param topk:
    :return: a dictionary: {'precision': pre_k, 'recall': rec_k, 'f1': f1_k}
    """
    top_idx_list = sorted(range(len(pred)), key=lambda i: pred[i])[-topk:]
    num_of_true_in_topk = len([idx for idx in top_idx_list if label[idx] == 1])
    # precision@k = #true label in topk / k
    pre_k = num_of_true_in_topk / float(topk)
    # recall@k = #true label in topk / #true label
    num_of_true_in_all = sum(label)
    if num_of_true_in_all > topk:
        rec_k = num_of_true_in_topk / float(topk)
    else:
        rec_k = num_of_true_in_topk / float(num_of_true_in_all)
    # f1@k = 2 * precision@k * recall@k / (precision@k + recall@k)
    if pre_k == 0 and rec_k == 0:
        f1_k = 0.0
    else:
        f1_k = 2 * pre_k * rec_k / (pre_k + rec_k)
    # return {'precision': pre_k, 'recall': rec_k, 'f1': f1_k}
    return pre_k, rec_k, f1_k


def evaluate_batch(pred, label, topk_list):
    pre = [0.0] * len(topk_list)
    rc = [0.0] * len(topk_list)
    f1 = [0.0] * len(topk_list)
    cnt = 0
    for i in range(0, pred.shape[0]):
        for idx, topk in enumerate(topk_list):
            pre_val, rc_val, f1_val = evaluate(pred=pred[i], label=label[i], topk=topk)
            pre[idx] += pre_val
            rc[idx] += rc_val
            f1[idx] += f1_val
        cnt += 1
    return pre, rc, f1, cnt


def evaluate_batch_f1_5(pred, label):
    f1_5_list = list()
    cnt = 0
    for i in range(0, pred.shape[0]):
        pre_5, rc_5, f1_5 = evaluate(pred=pred[i], label=label[i], topk=5)
        f1_5_list.append(f1_5)
        cnt += 1
    return f1_5_list, cnt
