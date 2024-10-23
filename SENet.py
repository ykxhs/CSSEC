import csv
import math
import os
import random

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from lifelines.statistics import multivariate_logrank_test
from tqdm import tqdm

from utils import AdoptiveSoftThreshold, p_normalize, regularizer, evaluate, cluster_evaluate

class MLP(nn.Layer):
    def __init__(self, in_dim, out_dim, hid_dim):
        # hid_dim should be a list
        super(MLP, self).__init__()
        self.layers = nn.LayerList()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        # 1st layer
        self.layers.append(nn.Linear(self.in_dim, self.hid_dim[0], weight_attr=nn.initializer.KaimingUniform()))
        self.layers.append(nn.ReLU())
        # hidden layer
        for i in range(len(self.hid_dim) - 1):
            self.layers.append(
                nn.Linear(self.hid_dim[i], self.hid_dim[i + 1], weight_attr=nn.initializer.KaimingUniform()))
            self.layers.append(nn.ReLU())
        # last layer
        self.out_layer = nn.Linear(self.hid_dim[-1], out_dim, weight_attr=nn.initializer.KaimingUniform())

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
        h = self.out_layer(h)
        h = paddle.tanh_(h)
        return h


class SENet(nn.Layer):
    def __init__(self, in_dim, out_dim, hid_dim):
        super(SENet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        # layers
        self.query_net = MLP(in_dim=self.in_dim, out_dim=self.out_dim, hid_dim=self.hid_dim)
        self.key_net = MLP(in_dim=self.in_dim, out_dim=self.out_dim, hid_dim=self.hid_dim)
        self.threshold = AdoptiveSoftThreshold(1)
        # hyparameters
        self.shrink = 1.0 / out_dim

    def query_embedding(self, query):
        query_emb = self.query_net(query)
        return query_emb

    def key_embedding(self, key):
        key_emb = self.key_net(key)
        return key_emb

    def get_coef(self, query, keys):
        c = self.threshold(paddle.mm(query, keys.T))
        return self.shrink * c

    def forward(self, x, others):
        query = self.query_embedding(x)
        key = self.key_embedding(others)
        out = self.get_coef(query, key)
        return out


def train_process(data, labels, conf, which):  # 自表达矩阵训练
    root_folder = "D://cyy/ASEFM"
    row_data = paddle.to_tensor(data.to_numpy(), dtype="float32")
    # row_data shape [20531, 1229]
    global_step = 0
    # aml 样本较少 修改 为 100
    for N in [200]:
        block_size = min(N, 600)
        sample_idx = np.random.choice(row_data.shape[1], N, replace=False)
        data = row_data.T[sample_idx]
        data = p_normalize(data)
        sample_shape = data.shape
        n_iter_per_epoch = data.shape[0] // conf["batch_size"]
        n_step_per_iter = round(data.shape[0] // block_size)
        n_epochs = conf["pre_epochs"] // n_iter_per_epoch

        local_senet = SENet(sample_shape[1], conf["out_dim"], conf["hid_dim"])
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=0.001)
        opt = paddle.optimizer.AdamW(learning_rate=conf["learning_rate"], parameters=local_senet.parameters())
        # opt = paddle.optimizer.AdamW(learning_rate=conf["learning_rate"], parameters=local_senet.parameters(),grad_clip=clip)
        scheduler = paddle.optimizer.lr.CosineAnnealingDecay(conf["learning_rate"], T_max=n_epochs,
                                                             eta_min=conf["min_lr"])

        n_iter = 0
        max_acc = 0.0
        pbar = tqdm(range(n_epochs), ncols=120)
        for epoch in pbar:
            pbar.set_description(f"SENet Epoch {epoch}")
            randidx = paddle.randperm(sample_shape[0])
            for i in range(n_iter_per_epoch):
                # each batch in sample
                local_senet.train()
                batch_idx = randidx[i * conf["batch_size"]: (i + 1) * conf["batch_size"]]
                batch = data[batch_idx]
                # process all embedding of query and key
                query_batch = local_senet.query_embedding(batch)
                key_batch = local_senet.query_embedding(batch)

                rec_batch = paddle.zeros_like(batch)
                reg = paddle.zeros([1])
                # each batch be reconstructed by sample
                for j in range(n_step_per_iter):
                    block = data[j * block_size: (j + 1) * block_size]
                    key_block = local_senet.key_embedding(block)
                    coef = local_senet.get_coef(query_batch, key_block)
                    rec_batch = rec_batch + paddle.mm(coef, block)
                    reg = reg + regularizer(coef, conf["lmbd"])

                diag_c = local_senet.threshold((query_batch * key_batch).sum(axis=1, keepdim=True)) * local_senet.shrink
                rec_batch = rec_batch - diag_c * batch
                rec_loss = paddle.sum(paddle.pow(batch - rec_batch, 2))
                loss = (0.5 * conf["gamma"] * rec_loss + reg) / conf["batch_size"]

                opt.clear_grad()
                loss.backward()
                opt.step()

                global_step += 1
                n_iter += 1

               
                if n_iter % conf["eval_iter"] == 0:
                    full_data = p_normalize(row_data)
                    coefficient, pred = evaluate(local_senet, data=full_data.T, num_subspaces=conf["subspace"],
                                                 affinity=conf["affinity"],
                                                 spectral_dim=conf["spectral_dim"], non_zeros=conf["non_zeros"],
                                                 n_neighbors=conf["n_neighbors"], batch_size=conf["chunk_size"],
                                                 chunk_size=conf["chunk_size"], knn_mode='symmetric')
                    simres = cluster_evaluate(labels,pred)
                    # nmi, ari, acc, pur,f_measure
                    if simres[2] > max_acc:
                        max_acc = simres[2]
                        print(simres)
                        paddle.save(local_senet.state_dict(), f"{root_folder}/sent_model/{which}_max_local.pdparams")

            pbar.set_postfix(loss="{:3.4f}".format(loss.item()),
                             rec_loss="{:3.4f}".format(rec_loss.item() / conf["batch_size"]),
                             reg="{:3.4f}".format(reg.item() / conf["batch_size"]))
            scheduler.step()

    return local_senet
