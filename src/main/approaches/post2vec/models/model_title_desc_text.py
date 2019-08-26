from torch.autograd import Variable
import torch.nn.functional as F
import os
import sys
import torch
import torch.nn as nn
from utils.eval_util import evaluate_batch, evaluate_batch_f1_5
import numpy as np
from utils.data_util import get_specific_comp_list
from utils.time_util import get_current_time


class MultiComp(nn.Module):

    def __init__(self, args):

        super(MultiComp, self).__init__()

        # args
        self.args = args
        t_V = args.title_embed_num
        dt_V = args.desc_text_embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        t_Ks = args.title_kernel_sizes
        dt_Ks = args.desc_text_kernel_sizes
        if len(t_Ks) == len(dt_Ks):
            num_of_Ks = len(t_Ks)
        else:
            print("kernel size not equal!")
            exit()

        self.title_embed = nn.Embedding(t_V, D)
        self.desc_text_embed = nn.Embedding(dt_V, D)
        self.convs_t = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in t_Ks])
        self.convs_dt = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in dt_Ks])
        self.dropout = nn.Dropout(args.dropout)

        self.fc1 = nn.Linear(num_of_Ks * Co * 2, C)

    def forward(self, t, dt):
        t = self.title_embed(t.cuda() if torch.cuda.is_available() else t)
        dt = self.desc_text_embed(dt.cuda() if torch.cuda.is_available() else dt)

        if self.args.static:
            t = Variable(t)
            dt = Variable(dt)

        t = t.unsqueeze(1)  # (N, Ci, W, D)
        dt = dt.unsqueeze(1)  # (N, Ci, W, D)

        t = [F.relu(conv(t)).squeeze(3) for conv in self.convs_t]  # [(N, Co, W), ...]*len(Ks)
        dt = [F.relu(conv(dt)).squeeze(3) for conv in self.convs_dt]  # [(N, Co, W), ...]*len(Ks)

        t = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in t]  # [(N, Co), ...]*len(Ks)
        dt = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in dt]  # [(N, Co), ...]*len(Ks)

        x_t, x_dt = torch.cat(t, 1), torch.cat(dt, 1)
        x = torch.cat((x_t, x_dt), 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        softmax = nn.Softmax()
        logit = self.fc1(x)  # (N, C)
        output = softmax(logit)

        return output

    def get_output_vector(self, qlist=None, batch_size=64):

        with torch.no_grad():
            for i in range(0, len(qlist), batch_size):
                st = i
                if i + batch_size < len(qlist):
                    et = i + batch_size
                else:
                    et = len(qlist)

                # features
                t = get_specific_comp_list("title", qlist[st:et])
                dt = get_specific_comp_list("desc_text", qlist[st:et])

                t = torch.tensor(t).long()
                dt = torch.tensor(dt).long()

                if self.args.cuda:
                    t, dt = t.cuda(), dt.cuda()

                t = self.title_embed(t.cuda() if torch.cuda.is_available() else t)
                dt = self.desc_text_embed(dt.cuda() if torch.cuda.is_available() else dt)

                if self.args.static:
                    t = Variable(t)
                    dt = Variable(dt)

                t = t.unsqueeze(1)  # (N, Ci, W, D)
                dt = dt.unsqueeze(1)  # (N, Ci, W, D)

                t = [F.relu(conv(t)).squeeze(3) for conv in self.convs_t]  # [(N, Co, W), ...]*len(Ks)
                dt = [F.relu(conv(dt)).squeeze(3) for conv in self.convs_dt]  # [(N, Co, W), ...]*len(Ks)

                t = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in t]  # [(N, Co), ...]*len(Ks)
                dt = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in dt]  # [(N, Co), ...]*len(Ks)

                x_t, x_dt = torch.cat(t, 1), torch.cat(dt, 1)

                for j in range(et - st):
                    qlist[st + j].title = (Variable(x_t).data).cpu().numpy()[j]
                    qlist[st + j].desc_text = (Variable(x_dt).data).cpu().numpy()[j]

        return qlist


def train(train_iter, dev_iter, model, args, global_train_step):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        print("\n#epoch %s" % epoch, get_current_time())
        for batch in train_iter:
            # features
            t = np.array(get_specific_comp_list("title", batch))
            dt = np.array(get_specific_comp_list("desc_text", batch))
            # label
            target = get_specific_comp_list("tags", batch)

            t = torch.tensor(t).long()
            dt = torch.tensor(dt).long()
            target = torch.tensor(target).float()

            if args.cuda:
                t, dt, target = t.cuda(), dt.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(t, dt)
            loss = nn.BCELoss()
            loss = loss(logit.reshape(-1), target.reshape(-1))

            loss.backward()
            optimizer.step()

            steps += 1
            global_train_step += 1
            if steps % args.log_interval == 0:
                sys.stdout.write('\rBatch[{}] - loss: {:.10f}'.format(steps, loss))
            if global_train_step % args.save_interval == 0:
                print("\nglobal_train_step {} - step {} - loss {:.10f}".format(global_train_step, steps, loss),
                      get_current_time())
                save(model, args.save_dir, 'snapshot', global_train_step)
    return model, global_train_step


def eval(data_iter, model, args, topk_list):
    with torch.no_grad():
        model.eval()
        pre = [0.0] * len(topk_list)
        rc = [0.0] * len(topk_list)
        f1 = [0.0] * len(topk_list)
        cnt = 0
        for batch in data_iter:
            # features
            t = get_specific_comp_list("title", batch)
            dt = get_specific_comp_list("desc_text", batch)
            # label
            target = np.array(get_specific_comp_list("tags", batch))

            t = torch.tensor(t).long()
            dt = torch.tensor(dt).long()
            target = torch.tensor(target).float()

            if args.cuda:
                t, dt, target = t.cuda(), dt.cuda(), target.cuda()

            logit = model(t, dt)
            if torch.cuda.is_available():
                pre_batch, rc_batch, f1_batch, cnt_batch = evaluate_batch(pred=logit.cpu().detach().numpy(),
                                                                          label=target.cpu().detach().numpy(),
                                                                          topk_list=topk_list)
            else:
                pre_batch, rc_batch, f1_batch, cnt_batch = evaluate_batch(pred=logit.detach().numpy(),
                                                                          label=target.detach().numpy(),
                                                                          topk_list=topk_list)

            for idx, topk in enumerate(topk_list):
                pre[idx] += pre_batch[idx]
                rc[idx] += rc_batch[idx]
                f1[idx] += f1_batch[idx]
            cnt += cnt_batch

    return pre, rc, f1, cnt


def eval_compute_f1_5(data_iter, model, args):
    with torch.no_grad():
        model.eval()

        f1_5_dict = dict()
        cnt = 0
        for batch in data_iter:

            id_list = [x.qid for x in batch]

            # features
            t = get_specific_comp_list("title", batch)
            dt = get_specific_comp_list("desc_text", batch)
            # label
            target = np.array(get_specific_comp_list("tags", batch))

            t = torch.tensor(t).long()
            dt = torch.tensor(dt).long()
            target = torch.tensor(target).float()

            if args.cuda:
                t, dt, target = t.cuda(), dt.cuda(), target.cuda()

            logit = model(t, dt)
            if torch.cuda.is_available():
                f1_batch, cnt_batch = evaluate_batch_f1_5(pred=logit.cpu().detach().numpy(),
                                                          label=target.cpu().detach().numpy())
            else:
                f1_batch, cnt_batch = evaluate_batch_f1_5(pred=logit.detach().numpy(),
                                                          label=target.detach().numpy())

            for i in range(len(id_list)):
                f1_5_dict[id_list[i]] = f1_batch[i]

    print("type of f1_5 %s" % type(f1_5_dict))
    return f1_5_dict, cnt


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # save parameter
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
