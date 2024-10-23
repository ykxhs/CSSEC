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
from sklearn.cluster import SpectralClustering
from tqdm import tqdm

import utils
from SEGN import SEGN
from SENet import train_process, SENet
from load_data import load_TCGA
warnings.filterwarnings("ignore")
DATASET_PATH = "D:/cyy/dataset/TCGA"
FLAGS_eager_delete_tensor_gb=0.0

def setup_seed(seed):
   random.seed(seed)
   os.environ["PYTHONHASHSEED"] = str(seed)
   np.random.seed(seed)
   paddle.seed(seed)

if __name__ == '__main__':
    seed = 123456 # 需要固定较大值
    setup_seed(seed)
    result_all = open("result/ASEFM.csv", 'w+')
    writer_all = csv.writer(result_all)
    writer_all.writerow(['log10p', 'dataset', 'cluster'])

    for cancer_type in ["aml","brca","coad","gbm","kirc","lihc","lusc","ov","sarc","skcm"]:
        # cancer_type = "lihc"
        [exp, methy, mirna, survival] = load_TCGA(DATASET_PATH,cancer_type,'knn_2')
        print(exp.shape,methy.shape,mirna.shape)
        # continue;
        conf = dict()
        conf["dataset"] = cancer_type
        conf["batch_size"] = 128
        conf["chunk_size"] = exp.shape[1]
        conf["out_dim"] = 512
        conf["hid_dim"] = [512]
        conf["pre_epochs"] = 2000
        conf['view'] = 3
        conf["learning_rate"] = 3e-4
        conf["lmbd"] = 0.95

        conf["gamma"] = 100
        conf["min_lr"] = 1e-5
        conf["save_iter"] = 500
        conf["eval_iter"] = 500
        conf["subspace"] = 5 #

        conf["spectral_dim"] = 15
        conf["non_zeros"] = 10000
        conf["n_neighbors"] = 5
        conf["affinity"] = "nearest_neighbor"

        # SENet
        exp_net = SENet(exp.shape[0], conf["out_dim"], conf["hid_dim"])
        methy_net = SENet(methy.shape[0], conf["out_dim"], conf["hid_dim"])
        mirna_net = SENet(mirna.shape[0], conf["out_dim"], conf["hid_dim"])
        trainSENet = False
        if trainSENet:
            exp_net = train_process(exp, survival, conf, "exp")
            methy_net = train_process(methy, survival, conf, "methy")
            mirna_net = train_process(mirna, survival, conf, "mirna")
        # make_graph
        conf['which_exp'] = 2000
        conf['which_methy'] = 2000
        conf['which_mirna'] = 2000
        para_dict = paddle.load(f"{conf['dataset']}_result/exp/exp_{conf['which_exp']}_local.pdparams")
        exp_net.load_dict(para_dict)
        para_dict = paddle.load(f"{conf['dataset']}_result/methy/methy_{conf['which_methy']}_local.pdparams")
        methy_net.load_dict(para_dict)
        para_dict = paddle.load(f"{conf['dataset']}_result/mirna/mirna_{conf['which_mirna']}_local.pdparams")
        mirna_net.load_dict(para_dict)
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
        utils.draw_coef(c2,pred2, conf['dataset'],"methy")
        utils.draw_coef(c3,pred3, conf['dataset'],"mirna")
        conf['make_graph_c1'] = 5
        conf['make_graph_c2'] = 5
        conf['make_graph_c3'] = 5
        g1, f1 = utils.make_graph(c1.toarray(), conf['make_graph_c1'])
        g2, f2 = utils.make_graph(c2.toarray(), conf['make_graph_c2'])
        g3, f3 = utils.make_graph(c3.toarray(), conf['make_graph_c3'])
        f1 = utils.p_normalize(f1)
        f2 = utils.p_normalize(f2)
        f3 = utils.p_normalize(f3)

        # SEGN
        conf["g_lmbd"] = 0.95
        conf["g_gamma"] = 100
        conf["g_con"] = 0.1
        conf["g_learning_rate"] = 1e-4
        conf['g_iter'] = 3000
        conf["eval_iter"] = 300

        g_dim = [f1.shape[0], f1.shape[0], 512]
        g_ae = SEGN(g_dim[0], g_dim[1], g_dim[2])

        folder = "{}_result".format(conf["dataset"])
        if not os.path.exists(folder):
            os.mkdir(folder)
        result = open(f'{folder}/graph_results.csv', 'w+')
        writer = csv.writer(result)
        writer.writerow(["-log10(p)","-log2(p)", "p", "iters"])
        # clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=0.001)
        # g_opt = paddle.optimizer.AdamW(learning_rate=conf["g_learning_rate"], parameters=g_ae.parameters(), grad_clip=clip)
        g_opt = paddle.optimizer.AdamW(learning_rate=conf["g_learning_rate"], parameters=g_ae.parameters())
        pbar = tqdm(range(conf['g_iter']))

        n_iter = 0
        max_pred = []
        max_log = 0.0
        loss_list = []
        for epoch in pbar:
            pbar.set_description(f"Global Epoch {epoch}")
            g_ae.train()
            coef, shared, emb, att, h_persudo = g_ae([g1, g2, g3], [f1, f2, f3])
            diag_c = g_ae.threshold((shared * emb).sum(axis=1, keepdim=True)) * g_ae.shrink
            rec_g1 = paddle.mm(coef, f1) - diag_c * f1
            rec_g2 = paddle.mm(coef, f2) - diag_c * f2
            rec_g3 = paddle.mm(coef, f3) - diag_c * f3
            reg = utils.regularizer(coef, conf["g_lmbd"])
            rec_loss = paddle.sum(paddle.pow(rec_g1 - f1, 2)) + paddle.sum(paddle.pow(rec_g2 - f2, 2)) + paddle.sum(paddle.pow(rec_g3 - f3, 2))
            loss = (0.5 * conf["g_gamma"] * rec_loss + reg) / f1.shape[0]

            consis_loss = paddle.sum(paddle.pow(paddle.matmul(shared.T, shared) - paddle.matmul(emb.T, emb), 2)) / f1.shape[0]
            loss += consis_loss
            temp = emb - paddle.unsqueeze(emb, axis=1)
            disparity_loss = 0.5 * paddle.multiply(paddle.pow(temp.sum(axis=2), 2),diag_c).sum() / f1.shape[0]
            loss += conf["g_con"] * disparity_loss

            g_opt.clear_grad()
            loss.backward()
            g_opt.step()

            n_iter += 1

            if n_iter % conf["save_iter"] == 0:
                paddle.save(g_ae.state_dict(), f"{conf['dataset']}_result/global/global_{n_iter}_local.pdparams")
                paddle.save(g_opt.state_dict(), "g_opt.pdopt")

            if n_iter % conf["eval_iter"] == 0:
                coefficient, pred = utils.g_evaluate(g_ae, [g1, g2, g3], [f1, f2, f3], num_subspaces=conf["subspace"],
                                                     affinity=conf["affinity"],
                                                     spectral_dim=conf["spectral_dim"], non_zeros=conf["non_zeros"],
                                                     n_neighbors=conf["n_neighbors"], batch_size=conf["chunk_size"],
                                                     chunk_size=conf["chunk_size"], knn_mode='symmetric')
                survival["label"] = pred
                utils.draw_coef(coefficient, pred, conf['dataset'],f"g_ae")
                df = survival
                results = multivariate_logrank_test(df['Survival'], df['label'], df['Death'])
                if results.summary["p"].item() < 1e-3:
                    paddle.save(g_ae.state_dict(), f"{conf['dataset']}_result/global/global_{n_iter}_local.pdparams")

                p = results.summary["p"].item()
                if -math.log10(p) > max_log:
                    max_log = -math.log10(p)
                    max_pred = pred

                writer.writerow([-math.log10(p), -math.log2(p), p, n_iter])
                result.flush()
            loss_list.append(loss.item())
            pbar.set_postfix(loss="{:3.4f}".format(loss.item()),
                             rec_loss="{:3.4f}".format(rec_loss.item() / f1.shape[0]),
                             reg="{:3.4f}".format(reg.item() / f1.shape[0]),
                             disparity_loss="{:3.4f}".format(disparity_loss.item())
                             )

        ## 输出结果
        plt.figure()
        plt.plot(loss_list)
        plt.savefig("./result/loss.png")
        print("-----------------res----------------")
        survival["label"] = np.array(max_pred)
        clinical_data = utils.get_clinical(DATASET_PATH + "/clinical", survival, conf["dataset"])
        cnt = utils.clinical_enrichement(clinical_data['label'],clinical_data)

        temp = pd.DataFrame(columns=['PatientID','labels'])
        temp['PatientID'] = survival['PatientID']
        temp['labels'] = max_pred
        temp.to_csv("./result/{}_{}".format("{}".format(conf['dataset']), "AMSCM"), index=False)
        fw = open("./parameter/{}_parameter.txt".format(conf['dataset']), 'w+')
        fw.write(str(conf))  # 把字典转化为str
        fw.close()

        print("{}:    AMSCM:  {:.2f}({})".format(conf['dataset'],max_log,len(set(max_pred))))
        writer_all.writerow(["{:.2f}".format(max_log),conf['dataset'],len(set(max_pred))])
        result_all.flush()
