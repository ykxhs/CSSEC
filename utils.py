import difflib
import math
import re

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import median_survival_times
from scipy.stats import chi2_contingency, kruskal
from sklearn import cluster

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pair_confusion_matrix, accuracy_score
from sklearn.preprocessing import normalize
import pgl
from snf import compute
from scipy.optimize import linear_sum_assignment

from matplotlib import pyplot as plt
from scipy import sparse
from sklearn.neighbors import kneighbors_graph
from sklearn.utils import check_symmetric, check_random_state, check_array


def regularizer(c, lmbd=1.0):
    # 公式7
    return lmbd * paddle.sum(paddle.abs(c)) + (1.0 - lmbd) / 2.0 * paddle.sum(paddle.pow(c, 2))

def p_normalize(x, p=2):
    return x / (paddle.norm(x, p=p, axis=1, keepdim=True) + 1e-6)

class AdoptiveSoftThreshold(nn.Layer):
    def __init__(self, dim):
        super(AdoptiveSoftThreshold, self).__init__()
        self.dim = dim
        self.add_parameter("bias", paddle.create_parameter(shape=[self.dim], dtype="float32",
                                                           attr=nn.initializer.Constant(value=0.0)))

    def forward(self, c):
        return paddle.sign(c) * F.relu(paddle.abs(c) - self.bias)


def make_graph(array, k=20):
    feature = paddle.to_tensor(array, dtype="float32")
    _, idx = paddle.topk(feature, k)
    idx = idx.numpy()
    edges = list()
    for i in range(idx.shape[0]):
        for j in idx[i]:
            edges.append((i, j))
    g = pgl.Graph(edges = edges,num_nodes = feature.shape[0],node_feat = {'nfeat':feature.T})
    return g.tensor(), feature


def get_knn_Aff(C_sparse_normalized, k=3, mode='symmetric'):
    C_knn = kneighbors_graph(C_sparse_normalized, k, mode='connectivity', include_self=False, n_jobs=10)
    if mode == 'symmetric':
        Aff_knn = 0.5 * (C_knn + C_knn.T)
    elif mode == 'reciprocal':
        Aff_knn = C_knn.multiply(C_knn.T)
    else:
        raise Exception("Mode must be 'symmetric' or 'reciprocal'")
    return Aff_knn


def spectral_clustering(affinity_matrix_, n_clusters, k, seed=123, n_init=20):
    affinity_matrix_ = check_symmetric(affinity_matrix_)
    random_state = check_random_state(seed)

    laplacian = sparse.csgraph.laplacian(affinity_matrix_, normed=True)
    _, vec = sparse.linalg.eigsh(sparse.identity(laplacian.shape[0]) - laplacian,
                                 k=k, sigma=None, which='LA')
    embedding = normalize(vec)
    _, labels_, _ = cluster.k_means(embedding, n_clusters,
                                    random_state=seed, n_init=n_init)
    return labels_


def get_sparse_rep(model, data, batch_size=10, chunk_size=100, non_zeros=10000):
    N, D = data.shape
    non_zeros = min(N, non_zeros)
    c = paddle.empty([batch_size, N])
    if (N % batch_size != 0):
        raise Exception("batch_size should be a factor of dataset size.")
    if (N % chunk_size != 0):
        raise Exception("chunk_size should be a factor of dataset size.")

    val = list()
    indicies = list()

    with paddle.no_grad():
        model.eval()
        for i in range(data.shape[0] // batch_size):
            chunk = data[i * batch_size:(i + 1) * batch_size].cuda()
            q = model.query_embedding(chunk)
            for j in range(data.shape[0] // chunk_size):
                chunk_samples = data[j * chunk_size: (j + 1) * chunk_size].cuda()
                k = model.key_embedding(chunk_samples)
                coef = model.get_coef(q, k)
                c[:, j * chunk_size:(j + 1) * chunk_size] = coef.cpu()

            # diag c reset to zero
            rows = list(range(batch_size))
            cols = [j + i * batch_size for j in rows]
            c[rows, cols] = 0.0
            tmp = paddle.zeros_like(c)
            # sort
            _, index = paddle.topk(paddle.abs(c), axis=1, k=non_zeros)
            for line in range(index.shape[0]):
                tmp[line] = c[line].gather(index[line])
            # val.append(paddle.gather(c, index=index, axis=1).reshape([-1]).cpu().numpy())
            val.append(tmp.reshape([-1]).cpu().numpy())
            index = index.reshape([-1]).cpu().numpy()
            indicies.append(index)

    val = np.concatenate(val, axis=0)
    indicies = np.concatenate(indicies, axis=0)
    indptr = [non_zeros * i for i in range(N + 1)]

    C_sparse = sparse.csr_matrix((val, indicies, indptr), shape=[N, N])
    return C_sparse


def evaluate(senet, data, num_subspaces, spectral_dim, non_zeros=1000, n_neighbors=3,
             batch_size=10000, chunk_size=10000, affinity='nearest_neighbor', knn_mode='symmetric'):
    # 输入网络和数据 返回亲和矩阵和谱聚类结果

    C_sparse = get_sparse_rep(model=senet, data=data, batch_size=batch_size,
                              chunk_size=chunk_size, non_zeros=non_zeros)
    C_sparse_normalized = normalize(C_sparse).astype(np.float32)
    # plt.matshow(np.sort(np.abs(C_sparse_normalized.toarray())))
    if affinity == 'symmetric':
        Aff = 0.5 * (np.abs(C_sparse_normalized) + np.abs(C_sparse_normalized).T)
    elif affinity == 'nearest_neighbor':
        Aff = get_knn_Aff(C_sparse_normalized, k=n_neighbors, mode=knn_mode)
    else:
        raise Exception("affinity should be 'symmetric' or 'nearest_neighbor'")
    fused = (Aff + Aff.T) / 2
    # first, second = get_n_clusters(np.array(fused.todense()))
    # n_cluster = first
    # print("SENet========{},{}".format(first, second))

    preds = spectral_clustering(Aff, num_subspaces, spectral_dim)

    return C_sparse_normalized, preds

def draw_coef(coefficient, pred, which = "./", name="vis"):
    pc = pd.DataFrame(np.abs(coefficient.toarray()))
    pc = pc + pc.T
    pc["label"] = pred
    pc = pc.sort_values("label")
    idx = pc.index # 排序之后原来的行序号nishu
    pc = pc[idx] # 把列也按序号排好 对齐标签
    plt.pcolor(pc)
    # plt.show()
    plt.savefig("./ASEFM/figs/{}_{}.png".format(which,name))


def g_evaluate(model, graphs, features, num_subspaces, spectral_dim, non_zeros=1000, n_neighbors=3,
               batch_size=10000, chunk_size=10000, affinity='nearest_neighbor', knn_mode='symmetric'):
    N, D = features[0].shape
    non_zeros = min(N, non_zeros)
    c = paddle.empty([batch_size, N])
    val = list()
    indicies = list()

    with paddle.no_grad():
        model.eval()
        coef, shared, emb, att, _ = model(graphs, features)
        c = coef.cpu()

        rows = list(range(batch_size))
        cols = [j for j in rows]
        c[rows, cols] = 0.0
        tmp = paddle.zeros_like(c)
        # sort
        _, index = paddle.topk(paddle.abs(c), axis=1, k=batch_size)
        for line in range(index.shape[0]):
            tmp[line] = c[line].gather(index[line])
        # val.append(paddle.gather(c, index=index, axis=1).reshape([-1]).cpu().numpy())
        val.append(tmp.reshape([-1]).cpu().numpy())
        index = index.reshape([-1]).cpu().numpy()
        indicies.append(index)
    val = np.concatenate(val, axis=0)
    indicies = np.concatenate(indicies, axis=0)
    indptr = [non_zeros * i for i in range(N + 1)]
    C_sparse = sparse.csr_matrix((val, indicies, indptr), shape=[N, N])

    C_sparse_normalized = normalize(C_sparse).astype(np.float32)
    # plt.matshow(np.sort(np.abs(C_sparse_normalized.toarray())))
    if affinity == 'symmetric':
        Aff = 0.5 * (np.abs(C_sparse_normalized) + np.abs(C_sparse_normalized).T)
    elif affinity == 'nearest_neighbor':
        Aff = get_knn_Aff(C_sparse_normalized, k=n_neighbors, mode=knn_mode)
    else:
        raise Exception("affinity should be 'symmetric' or 'nearest_neighbor'")

    # fused = (Aff + Aff.T) / 2
    # first, second = get_n_clusters(np.array(fused.todense()))
    # print("GENet========{},{}".format(first, second))
    # n_cluster = first

    preds = spectral_clustering(Aff, num_subspaces, spectral_dim)

    return C_sparse_normalized, preds

def lifeline_analysis(df, title_g="brca"):
    '''
    :param df:
    生存分析画图，传入参数为df是一个DataFrame
    拥有字段：label（预测对标签） Survival（生存时间） Death（是否死亡）
    :param title_g: 图标题
    :return:
    '''
    n_groups = len(set(df["label"]))
    kmf = KaplanMeierFitter()
    plt.figure()
    for group in range(n_groups):
        idx = (df["label"] == group)
        kmf.fit(df['Survival'][idx], df['Death'][idx], label='class_' + str(group))

        ax = kmf.plot()
        plt.title(title_g)
        plt.xlabel("lifeline(days)")
        plt.ylabel("survival probability")
        treatment_median_confidence_interval_ = median_survival_times(kmf.confidence_interval_)
    plt.show()


# 富集分析
def clinical_enrichement(label,clinical):
    cnt = 0
    # age 连续 使用KW检验
    # print(label,clinical)
    stat, p_value_age = kruskal(np.array(clinical["age"]), np.array(label))
    if p_value_age < 0.05:
        cnt += 1
        print("---age---")
    # 其余离散 卡方检验
    stat_names = ["gender","pathologic_T","pathologic_M","pathologic_N","pathologic_stage"]
    for stat_name in stat_names:
        if stat_name in clinical:
            c_table = pd.crosstab(clinical[stat_name],label,margins = True)
            stat, p_value_other, dof, expected = chi2_contingency(c_table)
            if p_value_other < 0.05:
                cnt += 1
                print(f"---{stat_name}---")
    return cnt


def log_rank(df):
    '''
    :param df: 传入生存数据
    拥有字段：label（预测对标签） Survival（生存时间） Death（是否死亡）
    :return: res 包含了p log2p log10p
    '''
    res = dict()
    results = multivariate_logrank_test(df['Survival'], df['label'], df['Death'])
    res['p'] = results.summary['p'].item()
    res['log10p'] = -math.log10(results.summary['p'].item())
    res['log2p'] = -math.log2(results.summary['p'].item())
    return res

def get_clinical(path,survival,cancer_type):
    clinical = pd.read_csv(f"{path}/{cancer_type}",sep="\t")
    if cancer_type == 'kirc':
        replace = {'gender.demographic': 'gender','submitter_id.samples': 'sampleID'}
        clinical = clinical.rename(columns=replace)  # 为某个 index 单独修改名称
        clinical["sampleID"] = [re.sub("A", "", x) for x in clinical["sampleID"].str.upper()]
    clinical["sampleID"] = [re.sub("-", ".", x) for x in clinical["sampleID"].str.upper()]
    survival['age'] = pd.NA # 初始化年龄
    survival['gender'] = pd.NA # 初始化年龄
    if 'pathologic_T' in clinical.columns:
        survival['T'] = pd.NA # 初始化年龄
    if 'pathologic_M' in clinical.columns:
        survival['M'] = pd.NA # 初始化年龄
    if 'pathologic_N' in clinical.columns:
        survival['N'] = pd.NA # 初始化年龄
    if 'tumor_stage.diagnoses' in clinical.columns:
        survival['stage'] = pd.NA # 初始化年龄
    i = 0
    # 找对应的参数
    for name in survival['PatientID']:
        # print(name)
        flag = difflib.get_close_matches(name,list(clinical["sampleID"]),1,cutoff=0.6)
        if flag:
            idx = list(clinical["sampleID"]).index(flag[0])
            survival['age'][i] = clinical['age_at_initial_pathologic_diagnosis'][idx]
            survival['gender'][i] = clinical['gender'][idx]
            if 'pathologic_T' in clinical.columns:
                survival['T'][i] = clinical['pathologic_T'][idx]
            if 'pathologic_M' in clinical.columns:
                survival['M'][i] = clinical['pathologic_M'][idx]
            if 'pathologic_N' in clinical.columns:
                survival['N'][i] = clinical['pathologic_N'][idx]
            if 'tumor_stage.diagnoses' in clinical.columns:
                survival['stage'][i] = clinical['tumor_stage.diagnoses'][idx]
        else: print(name)
        i = i + 1
    return survival.dropna(axis=0, how='any')


def get_n_clusters(arr, n_clusters=range(2, 6)):
    """
    Finds optimal number of clusters in `arr` via eigengap method

    Parameters
    ----------
    arr : (N, N) array_like
        Input array (e.g., the output of :py:func`snf.compute.snf`)
    n_clusters : array_like
        Numbers of clusters to choose between

    Returns
    -------
    opt_cluster : int
        Optimal number of clusters
    second_opt_cluster : int
        Second best number of clusters
    """

    # confirm inputs are appropriate
    n_clusters = check_array(n_clusters, ensure_2d=False)
    n_clusters = n_clusters[n_clusters > 1]
    # don't overwrite provided array!
    graph = arr.copy()
    graph = (graph + graph.T) / 2
    graph[np.diag_indices_from(graph)] = 0
    degree = graph.sum(axis=1)
    degree[np.isclose(degree, 0)] += np.spacing(1)
    di = np.diag(1 / np.sqrt(degree))
    laplacian = di @ (np.diag(degree) - graph) @ di

    # perform eigendecomposition and find eigengap
    eigs = np.sort(np.linalg.eig(laplacian)[0])
    eigengap = np.abs(np.diff(eigs))
    eigengap = eigengap * (1 - eigs[:-1]) / (1 - eigs[1:])
    n = eigengap[n_clusters - 1].argsort()[::-1]

    return n_clusters[n[:2]]


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def get_rand_index_and_f_measure(labels_true, labels_pred, beta=1.):
     (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
     ri = (tp + tn) / (tp + tn + fp + fn)
     ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
     p, r = tp / (tp + fp), tp / (tp + fn)
     f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
     return ri, ari, f_beta

def cluster_evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    f_measure = get_rand_index_and_f_measure(label,pred)[2]
    return nmi, ari, acc, pur,f_measure

