import csv
import math
import os
import random
import warnings

import numpy as np
import paddle
import pandas as pd
from lifelines.statistics import multivariate_logrank_test
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from tqdm import tqdm

import utils
from SEGN import SEGN
from SENet import train_process, SENet
from load_data import load_TCGA
warnings.filterwarnings("ignore")
DATASET_PATH = "./ASEFM/data/"
FLAGS_eager_delete_tensor_gb=0.0

def setup_seed(seed):
   random.seed(seed)
   os.environ["PYTHONHASHSEED"] = str(seed)
   np.random.seed(seed)
   paddle.seed(seed)

if __name__ == '__main__':
    seed = 123456 # 需要固定较大值
    setup_seed(seed)
    for cancer_type in ["simdata3","simdata2","simdata3"]:
        result_all = open(f"D://cyy/ASEFM/result/ASEFM_{cancer_type}.csv", 'w+')
        writer_all = csv.writer(result_all)
        writer_all.writerow(['ACC', 'NMI', 'F_beta'])

        for times in range(50):
            exp = pd.read_csv(DATASET_PATH + cancer_type + f"/exp_{times}.csv",header=None)
            methy = pd.read_csv(DATASET_PATH + cancer_type + f"/methy_{times}.csv",header=None)
            mirna = pd.read_csv(DATASET_PATH + cancer_type + f"/mirna_{times}.csv",header=None)
            omics_list = pd.concat([exp,methy,mirna])
            print(omics_list.shape)

            print(exp.shape,methy.shape,mirna.shape)
            conf = dict()
            conf["dataset"] = cancer_type
            conf["batch_size"] = 128
            conf["chunk_size"] = exp.shape[1]
            conf["out_dim"] = 128
            conf["hid_dim"] = [512]
            conf["pre_epochs"] = 2000
            conf['view'] = 3
            conf["learning_rate"] = 3e-4
            conf["lmbd"] = 0.9

            conf["gamma"] = 10
            conf["min_lr"] = 1e-5
            conf["save_iter"] = 500
            # conf["eval_iter"] = 50
            conf["eval_iter"] = conf["pre_epochs"]
            conf["subspace"] = 3 #

            conf["spectral_dim"] = 15
            conf["non_zeros"] = 10000
            conf["n_neighbors"] = 5
            conf["affinity"] = "nearest_neighbor"

            labels = np.array([int(x / (exp.shape[1] / conf["subspace"])) + 1 for x in range(exp.shape[1])])
            print(labels)
            kmeans_pred = KMeans(n_clusters=conf['subspace'], random_state=0).fit_predict(exp.T)
            simres = utils.cluster_evaluate(labels,kmeans_pred)
            print(simres)



            # SENet
            exp_net = SENet(exp.shape[0], conf["out_dim"], conf["hid_dim"])
            methy_net = SENet(methy.shape[0], conf["out_dim"], conf["hid_dim"])
            mirna_net = SENet(mirna.shape[0], conf["out_dim"], conf["hid_dim"])
            trainSENet = True
            if trainSENet:
                print("==============Train SENet exp========================")
                exp_net = train_process(exp, labels, conf, "exp")
                print("==============Train SENet methy========================")
                methy_net = train_process(methy, labels, conf, "methy")
                print("==============Train SENet mirna========================")
                mirna_net = train_process(mirna, labels, conf, "mirna")
            
            
            # # # make_graph
            # # conf['which_exp'] = 2000
            # # conf['which_methy'] = 2000
            # # conf['which_mirna'] = 2000
            # # para_dict = paddle.load(f"{conf['dataset']}_result/exp/exp_{conf['which_exp']}_local.pdparams")
            # # exp_net.load_dict(para_dict)
            # # para_dict = paddle.load(f"{conf['dataset']}_result/methy/methy_{conf['which_methy']}_local.pdparams")
            # # methy_net.load_dict(para_dict)
            # # para_dict = paddle.load(f"{conf['dataset']}_result/mirna/mirna_{conf['which_mirna']}_local.pdparams")
            # # mirna_net.load_dict(para_dict)
            c1, pred1 = utils.evaluate(exp_net, data=utils.p_normalize(paddle.to_tensor(exp.to_numpy().T, dtype="float32")), num_subspaces=conf["subspace"], affinity=conf["affinity"],
                                    spectral_dim=conf["spectral_dim"], non_zeros=conf["non_zeros"], n_neighbors=conf["n_neighbors"], batch_size=conf["chunk_size"],
                                    chunk_size=conf["chunk_size"], knn_mode='symmetric')
            c2, pred2 = utils.evaluate(methy_net, data=utils.p_normalize(paddle.to_tensor(methy.to_numpy().T, dtype="float32")), num_subspaces=conf["subspace"], affinity=conf["affinity"],
                                        spectral_dim=conf["spectral_dim"], non_zeros=conf["non_zeros"], n_neighbors=conf["n_neighbors"], batch_size=conf["chunk_size"],
                                        chunk_size=conf["chunk_size"], knn_mode='symmetric')
            c3, pred3 = utils.evaluate(mirna_net, data=utils.p_normalize(paddle.to_tensor(mirna.to_numpy().T, dtype="float32")), num_subspaces=conf["subspace"], affinity=conf["affinity"],
                                        spectral_dim=conf["spectral_dim"], non_zeros=conf["non_zeros"], n_neighbors=conf["n_neighbors"], batch_size=conf["chunk_size"],
                                    chunk_size=conf["chunk_size"], knn_mode='symmetric')
            utils.draw_coef(c1, pred1, conf['dataset'],"exp")
            utils.draw_coef(c2, pred2, conf['dataset'],"methy")
            utils.draw_coef(c3, pred3, conf['dataset'],"mirna") 


            # conf['make_graph_c1'] = 5
            # conf['make_graph_c2'] = 5
            # conf['make_graph_c3'] = 5
            # g1, f1 = utils.make_graph(c1.toarray(), conf['make_graph_c1'])
            # g2, f2 = utils.make_graph(c2.toarray(), conf['make_graph_c2'])
            # g3, f3 = utils.make_graph(c3.toarray(), conf['make_graph_c3'])
            # f1 = utils.p_normalize(f1)
            # f2 = utils.p_normalize(f2)
            # f3 = utils.p_normalize(f3)

            # # SEGN
            # conf["g_lmbd"] = 0.5
            # conf["g_gamma"] = 0.1
            # conf["g_con"] = 0.9
            # conf["g_learning_rate"] = 3e-4
            # conf['g_iter'] = 500
            # conf["eval_iter"] = 100

            # g_dim = [f1.shape[0], f1.shape[0], 512]
            # g_ae = SEGN(g_dim[0], g_dim[1], g_dim[2])
            # # clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=0.001)
            # # g_opt = paddle.optimizer.AdamW(learning_rate=conf["g_learning_rate"], parameters=g_ae.parameters(), grad_clip=clip)
            # g_opt = paddle.optimizer.AdamW(learning_rate=conf["g_learning_rate"], parameters=g_ae.parameters())
            # pbar = tqdm(range(conf['g_iter']))

            # n_iter = 0
            # max_sim = [0.0,0.0,0.0,0.0,0.0]
            # for epoch in pbar:
            #     pbar.set_description(f"Global Epoch {epoch}")
            #     g_ae.train()
            #     coef, shared, emb, att, h_persudo = g_ae([g1, g2, g3], [f1, f2, f3])
            #     diag_c = g_ae.threshold((shared * emb).sum(axis=1, keepdim=True)) * g_ae.shrink
            #     rec_g1 = paddle.mm(coef, f1) - diag_c * f1
            #     rec_g2 = paddle.mm(coef, f2) - diag_c * f2
            #     rec_g3 = paddle.mm(coef, f3) - diag_c * f3
            #     reg = utils.regularizer(coef, conf["g_lmbd"])
            #     rec_loss = paddle.sum(paddle.pow(rec_g1 - f1, 2)) + paddle.sum(paddle.pow(rec_g2 - f2, 2)) + paddle.sum(paddle.pow(rec_g3 - f3, 2))
            #     loss = (0.5 * conf["g_gamma"] * rec_loss + reg) / f1.shape[0]

            #     consis_loss = paddle.sum(paddle.pow(paddle.matmul(shared.T, shared) - paddle.matmul(emb.T, emb), 2)) / f1.shape[0]
            #     loss += consis_loss
            #     temp = emb - paddle.unsqueeze(emb, axis=1)
            #     disparity_loss = 0.5 * paddle.multiply(paddle.pow(temp.sum(axis=2), 2),diag_c).sum() / f1.shape[0]
            #     loss += conf["g_con"] * disparity_loss

            #     g_opt.clear_grad()
            #     loss.backward()
            #     g_opt.step()

            #     n_iter += 1

            #     if n_iter % conf["eval_iter"] == 0:
            #         coefficient, pred = utils.g_evaluate(g_ae, [g1, g2, g3], [f1, f2, f3], num_subspaces=conf["subspace"],
            #                                             affinity=conf["affinity"],
            #                                             spectral_dim=conf["spectral_dim"], non_zeros=conf["non_zeros"],
            #                                             n_neighbors=conf["n_neighbors"], batch_size=conf["chunk_size"],
            #                                             chunk_size=conf["chunk_size"], knn_mode='symmetric')
            #         simres = utils.cluster_evaluate(labels,pred)
            #         # nmi, ari, acc, pur,f_measure
            #         if simres[2] > max_sim[2]:
            #             print(simres)
            #             max_sim = simres
                        
            #     pbar.set_postfix(loss="{:3.4f}".format(loss.item()),
            #                     rec_loss="{:3.4f}".format(rec_loss.item() / f1.shape[0]),
            #                     reg="{:3.4f}".format(reg.item() / f1.shape[0]),
            #                     disparity_loss="{:3.4f}".format(disparity_loss.item())
            #                     )


            # writer_all.writerow([max_sim[0],max_sim[1],max_sim[2],max_sim[3],max_sim[4]])
            # result_all.flush()
