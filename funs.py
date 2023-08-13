import os
import time
import random
import numpy as np
import pandas as pd
import xlrd
import scipy
import scipy.io as sio
from scipy import stats
from scipy import sparse

from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.metrics import adjusted_rand_score as ari_ori
from sklearn.metrics import adjusted_mutual_info_score as ami_ori
from sklearn.metrics import normalized_mutual_info_score as nmi_ori

from sklearn.metrics.pairwise import euclidean_distances as EuDist2
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
from multiprocessing import Pool
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.transforms as mt
from sklearn.decomposition import PCA
import pandas as pd

import selectedImportantFeatures


def get_anchor(X, m, way="random"):
    if way == "kmeans":
        A = KMeans(m, init='random').fit(X).cluster_centers_
    elif way == "kmeans2":
        A = KMeans(m, init='random').fit(X).cluster_centers_
        D = EuDist2(A, X)
        ind = np.argmin(D, axis=1)
        A = X[ind, :]
    elif way == "k-means++":
        A = KMeans(m, init='k-means++').fit(X).cluster_centers_
    elif way == "k-means++2":
        A = KMeans(m, init='k-means++').fit(X).cluster_centers_
        D = EuDist2(A, X)
        A = np.argmin(D, axis=1)
    elif way == "random":
        ids = random.sample(range(X.shape[0]), m)
        A = X[ids, :]
    return A


def precision(y_true, y_pred):
    assert (len(y_pred) == len(y_true))
    N = len(y_pred)
    y_df = pd.DataFrame(data=y_pred, columns=["label"])
    ind_L = y_df.groupby("label").indices
    ni_L = [stats.mode(y_true[ind]).count[0] for yi, ind in ind_L.items()]
    return np.sum(ni_L) / N


def recall(y_true, y_pred):
    re = precision(y_true=y_pred, y_pred=y_true)
    return re


def accuracy(y_true, y_pred):
    """Get the best accuracy.

    Parameters
    ----------
    true_row_labels: array-like
        The true row labels, given as external information
    predicted_row_labels: array-like
        The row labels predicted by the model

    Returns
    -------
    float
        Best value of accuracy
    """

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cost_m = np.max(cm) - cm
    indices = linear_sum_assignment(cost_m)
    indices = np.asarray(indices)
    indexes = np.transpose(indices)
    total = 0
    for row, column in indexes:
        value = cm[row][column]
        total += value
    return total * 1. / np.sum(cm)


def fmi(y_true, y_pred):
    ret = metrics.fowlkes_mallows_score(labels_true=y_true, labels_pred=y_pred)
    return ret


def ari(y_true, y_pred):
    ret = ari_ori(labels_true=y_true, labels_pred=y_pred)
    return ret


def ami(y_true, y_pred, average_method="max"):
    ret = ami_ori(labels_true=y_true, labels_pred=y_pred, average_method=average_method)
    return ret


def nmi(y_true, y_pred, average_method="max"):
    ret = nmi_ori(labels_true=y_true, labels_pred=y_pred, average_method=average_method)
    return ret


def load_Agg():
    this_directory = os.path.dirname(__file__)
    data_path = os.path.join(this_directory, "dataset/")
    name_full = os.path.join(data_path + "Agg.mat")
    X, y_true, N, dim, c_true = load_mat(name_full)
    return X, y_true, N, dim, c_true


def load_USPS():
    this_directory = os.path.dirname(__file__)
    data_path = os.path.join(this_directory, "dataset/")
    name_full = os.path.join(data_path + "USPS.mat")
    X, y_true, N, dim, c_true = load_mat(name_full)
    return X, y_true, N, dim, c_true


def load_mat(path):
    data = sio.loadmat(path)
    X = data["X"]
    if "y_true" in data.keys():
        y_true = data["y_true"].astype(np.int32).reshape(-1)
    elif "Y" in data.keys():
        y_true = data["Y"].astype(np.int32).reshape(-1)
    else:
        assert 1==0

    N, dim, c_true = X.shape[0], X.shape[1], len(np.unique(y_true))
    return X, y_true, N, dim, c_true


def save_mat(name_full, xy):
    sio.savemat(name_full, xy)

def load_csv(path,label='Y'):
    #转录组
    df = pd.read_csv(path, sep=r',', header=0, index_col=0)
    df = df.fillna(0)
    print(df)
    df=df.iloc[:,:]
    print(df)
    X = df.T
    print(X)
    N=X.shape[0]
    return X, N
    #单细胞(无特征选择）

    # df = pd.read_csv(path, sep=',', header=None, index_col=None).T
    # matrix=df.values
    # print(matrix)
    # X = matrix[1:, 0:]
    # print(X)
    # y_true = matrix[1:, 22085:].astype(np.int32).reshape(-1)
    # print(y_true)
    # N = X.shape[0]#, X.shape[1], len(np.unique(y_true))
    # return X, y_true, N
    # 单细胞(特征选择)
    # df = pd.read_csv(path, sep=',', header=None, index_col=None).T
    # matrix = df.values
    # print(matrix)
    # Z = matrix[1:, 0:22085]
    # print(Z)
    # y_true = matrix[1:, 22085:].astype(np.int32).reshape(-1)
    # print(y_true)
    # X=selectedImportantFeatures.importantFeatures(Z,y_true)
    # print(X)
    # N = X.shape[0]  # , X.shape[1], len(np.unique(y_true))
    # return X, y_true, N





def load_har(path):
    #x_train= (pd.read_csv(path + '/train/mouseX_train.txt', sep=r'\s+', header=None)).T
    #y_train = pd.read_csv(path + '/train/mouseY_train.txt', header=None)
    #x_test = (pd.read_csv(path + '/test/mouseX_text.txt', sep=r'\s+', header=None)).T
    #y_test = pd.read_csv(path + '/test/mouseY_text.txt', header=None)
    x_train = pd.read_csv(path+'/train/X_train.txt',sep=r'\s+',header=None)
    y_train = pd.read_csv(path+'/train/y_train.txt', header=None)
    x_test = pd.read_csv(path+'/test/X_test.txt', sep=r'\s+', header=None)
    y_test = pd.read_csv(path+'/test/y_test.txt', header=None)

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    # labels start at 1 so..
    y = y - 1
    y = y.reshape((y.size,))

    N, dim, c_true = x.shape[0], x.shape[1], len(np.unique(y))
    return x, y, N, dim, c_true

def load_pendigits(data_path='data/pendigits'):
    if not os.path.exists(data_path + '/pendigits.tra'):
        os.makedirs(data_path, exist_ok=True)

        os.system(
            'wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra -P %s' %
            data_path)
        os.system(
            'wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes -P %s' %
            data_path)
        os.system(
            'wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.names -P %s' %
            data_path)

    # load training data
    with open(data_path + '/pendigits.tra') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_train, labels_train = data[:, :-1], data[:, -1]

    # load testing data
    with open(data_path + '/pendigits.tes') as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_test, labels_test = data[:, :-1], data[:, -1]

    x = np.concatenate((data_train, data_test)).astype('float64')
    y = np.concatenate((labels_train, labels_test))
    x /= 100.
    y = y.astype('int')

    N, dim, c_true = x.shape[0], x.shape[1], len(np.unique(y))

    return x, y, N, dim, c_true

def load_usps(data_path='data/usps'):
    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64')
    y = np.concatenate((labels_train, labels_test))

    N, dim, c_true = x.shape[0], x.shape[1], len(np.unique(y))

    return x, y, N, dim, c_true

def matrix_index_take(X, ind_M):
    assert np.all(ind_M >= 0)

    n, k = ind_M.shape
    row = np.repeat(np.array(range(n), dtype=np.int32), k)
    col = ind_M.reshape(-1)
    ret = X[row, col].reshape((n, k))
    return ret


def matrix_index_assign(X, ind_M, Val):
    n, k = ind_M.shape
    row = np.repeat(np.array(range(n), dtype=np.int32), k)
    col = ind_M.reshape(-1)
    if isinstance(Val, (float, int)):
        X[row, col] = Val
    else:
        X[row, col] = Val.reshape(-1)


def EProjSimplex_new(v, k=1):
    v = v.reshape(-1)
    # min  || x- v ||^2
    # s.t. x>=0, sum(x)=k
    ft = 1
    n = len(v)
    v0 = v-np.mean(v) + k/n
    vmin = np.min(v0)

    if vmin < 0:
        f = 1
        lambda_m = 0
        while np.abs(f) > 1e-10:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = np.sum(posidx)
            g = -npos
            f = np.sum(v1[posidx]) - k
            lambda_m -= f/g
            ft += 1
            if ft > 100:
                x = np.maximum(v1, 0)
                break
        x = np.maximum(v1, 0)
    else:
        x = v0

    return x, ft


def EProjSimplexdiag(d, u):
    #  d = d.astype(np.float64)
    #  u = u.astype(np.float64)
    # min  1/2*x'*U*x - x'*d
    # s.t. x>=0, sum(x) = 1
    lam = np.min(u-d)
    #  print(lam)
    f = 1
    count = 1
    while np.abs(f) > 1e-8:
        v1 = (lam + d)/u
        posidx = v1 > 0
        #  print(v1)
        g = np.sum(1/u[posidx])
        f = np.sum(v1[posidx]) - 1
        #  print(f)
        lam -= f/g

        if count > 1000:
            break
        count += 1
    v1 = (lam+d)/u
    x = np.maximum(v1, 0)
    return x, f


def eig1(A, c, isMax=True, isSym=True):
    if isinstance(A, sparse.spmatrix):
        A = A.toarray()

    if isSym:
        A = np.maximum(A, A.T)

    if isSym:
        d, v = np.linalg.eigh(A)
    else:
        d, v = np.linalg.eig(A)

    if isMax:
        idx = np.argsort(-d)
    else:
        idx = np.argsort(d)

    idx1 = idx[:c]
    eigval = d[idx1]
    eigvec = v[:, idx1]

    eigval_full = d[idx]

    return eigvec, eigval, eigval_full


def knn_f(X, knn, squared=True):
    D_full = EuDist2(X, X, squared=squared)
    np.fill_diagonal(D_full, -1)
    NN_full = np.argsort(D_full, axis=1)
    np.fill_diagonal(D_full, 0)

    NN = NN_full[:, :knn]
    NND = matrix_index_take(D_full, NN)
    return NN, NND


def kng(X, knn, way="gaussian", t="mean", Anchor=0, isSym=True):
    """
    :param X: data matrix of n by d
    :param knn: the number of nearest neighbors
    :param way: one of ["gaussian", "t_free"]
        "t_free" denote the method proposed in :
            "The constrained laplacian rank algorithm for graph-based clustering"
        "gaussian" denote the heat kernel
    :param t: only needed by gaussian, the bandwidth parameter
    :param Anchor: Anchor set, m by d
    :return: A, an sparse matrix (graph) of n by n if Anchor = 0 (default)
    """
    N, dim = X.shape
    if isinstance(Anchor, int):
        # n x n graph
        D = EuDist2(X, X, squared=True)
        ind_M = np.argsort(D, axis=1)
        if way == "gaussian":
            Val = matrix_index_take(D, ind_M[:, 1:(knn+1)])
            if t == "mean":
                t = np.mean(Val)
            elif t == "median":
                t = np.median(Val)
            Val = np.exp(-Val / t)

        elif way == "t_free":
            Val = matrix_index_take(D, ind_M[:, 1:(knn+2)])
            Val = Val[:, knn].reshape((-1, 1)) - Val[:, :knn]
            ind0 = np.where(Val[:, 0] == 0)[0]
            if len(ind0) > 0:
                Val[ind0, :] = 1/knn
            Val = Val / np.sum(Val, axis=1).reshape(-1, 1)
        A = np.zeros((N, N))
        matrix_index_assign(A, ind_M[:, 1:(knn+1)], Val)
        if isSym:
            A = (A + A.T) / 2
    else:
        # n x m graph
        num_anchor = Anchor.shape[0]
        D = EuDist2(X, Anchor, squared=True)  # n x m
        ind_M = np.argsort(D, axis=1)
        if way == "gaussian":
            Val = matrix_index_take(D, ind_M[:, :knn])
            if t == "mean":
                t = np.mean(Val)
            elif t == "median":
                t = np.median(Val)
            Val = np.exp(-Val / t)
        elif way == "t_free":
            Val = matrix_index_take(D, ind_M[:, :(knn+1)])
            Val = Val[:, knn].reshape((-1, 1)) - Val[:, :knn]
            Val = Val / np.sum(Val, axis=1).reshape(-1, 1)
        A = np.zeros((N, num_anchor))
        matrix_index_assign(A, ind_M[:, :knn], Val)

    return A

# :param init = random or k - means + +


def kmeans(X, c, rep, init="random", mini_batch=False, n_cpu=6):
    '''
    :param X: 2D-array with size of N x dim
    :param c: the number of clusters to construct
    :param rep: the number of runs
    :param init: the way of initialization: random (default), k-means++
    :param mini_batch: mini_batch kmeans: True, False (default)
    :param par: parallel: True (default), False
    :param n_cpu: the number of cores used if par==True, n_cpu = 6, by default.
    :return: Y, 2D-array with size of rep x N, each row is a assignment
    '''

    Y = np.zeros((rep, X.shape[0]), dtype=np.int32)
    if mini_batch:
        for i in range(rep):
            Y[i, :] = MiniBatchKMeans(n_clusters=c, n_init=1, init=init).fit(X).predict(X)
    else:
        for i in range(rep):
            Y[i, :] =          KMeans(n_clusters=c, n_init=1, init=init).fit(X).labels_
    return Y


def relabel(y, offset=0):
    y_df = pd.DataFrame(data=y, columns=["label"])
    ind_dict = y_df.groupby("label").indices

    for yi, ind in ind_dict.items():
        y[ind] = offset
        offset += 1


def normalize_fea(fea, row):
    '''
    if row == 1, normalize each row of fea to have unit norm;
    if row == 0, normalize each column of fea to have unit norm;
    '''

    if 'row' not in locals():
        row = 1

    if row == 1:
        feaNorm = np.maximum(1e-14, np.sum(fea ** 2, 1).reshape(-1, 1))
        fea = fea / np.sqrt(feaNorm)
    else:
        feaNorm = np.maximum(1e-14, np.sum(fea ** 2, 0))
        fea = fea / np.sqrt(feaNorm)

    return fea


def plot_bar(Y, labels, fname, barWidth=0.3, ra=0.5, addy=10):
    plt.figure(figsize=(6.4, 4.8))

    d_external = (1-2*barWidth)/(1+ra)
    d_internal = ra*d_external

    # Set position of bar on X axis
    X = np.zeros_like(Y)
    x_base = np.arange(Y.shape[0])
    X[:, 0] = x_base - 0.5*d_internal - 0.5*barWidth
    X[:, 1] = x_base + 0.5*d_internal + 0.5*barWidth

    # Make the plot
    plt.bar(X[:, 0], Y[:, 0], color=[0.99, 0.5, 0.05], width=barWidth,
            edgecolor='white', label='Random')
    plt.bar(X[:, 1], Y[:, 1], color=[0.137, 0.533, 0.8], width=barWidth,
            edgecolor='white', label=r'$k$-means')

    for g in range(Y.shape[1]):
        for i in range(Y.shape[0]):
            plt.text(X[i, g], Y[i, g], str(Y[i, g]), rotation=0, horizontalalignment='center')

    plt.xticks(x_base, labels=labels)

    plt.ylim(np.minimum(np.min(Y), 0), np.max(Y) + addy)
    plt.legend(loc='upper left')
    plt.yticks([])
    plt.subplots_adjust(top=0.98, bottom=0.06, right=0.98, left=0.02, hspace=0, wspace=0)
    plt.margins(0.05, 0.05)

    plt.savefig(fname=fname, format="pdf")
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.savefig("/home/pei/Hm.C.pdf", dpi=300, bbox_inches=mt.Bbox([[-0.1, -0.1], [6.5, 4.9]]))
    plt.show()


def etaFunction(t, etaBase, rho, r):
    if t <= r:
        eta = etaBase * (t/r)**rho
    else:
        eta = etaBase

    return eta


def initialY(init, N, c_true=None, X=None):
    if isinstance(init, np.ndarray):
        y = init
    elif isinstance(init, str) and init == "random":
        y = np.random.randint(0, c_true, N)

    else:
        assert X is not None
        assert c_true is not None
        assert isinstance(init, str)

        if init == "kmeans":
            y = KMeans(n_clusters=c_true, init="random", n_init=1).fit(X).labels_
        elif init == "kmeans++":
            y = KMeans(n_clusters=c_true, init="k-means++", n_init=1).fit(X).labels_

    assert y.shape[0] == N

    return y

def data_description(data_path, data_name, version, url):
    X, y_true, N, dim, c_true = load_mat(
            "{}{}_{}.mat".format(data_path, data_name, version))

    # title and content
    T1 = "data_name"
    T2 = "# Samples"
    T3 = "# Features"
    T4 = "# Subjects"

    C1 = data_name
    C2 = str(X.shape[0])
    C3 = str(X.shape[1])
    C4 = str(c_true)

    n1 = max(len(T1), len(C1))
    n2 = max(len(T2), len(C2))
    n3 = max(len(T3), len(C3))
    n4 = max(len(T4), len(C4))

    y_df = pd.DataFrame(data=y_true, columns=["label"])
    ind_L = y_df.groupby("label").size()

    show_n = 5

    with open("{}{}_{}.txt".format(data_path, data_name, version), "a") as f:

        # version
        f.write("version = {}\n\n".format(version))

        # table
        f.write("{}  {}  {}  {}\n".format(
            T1.rjust(n1), T2.rjust(n2), T3.rjust(n3), T4.rjust(n4)))
        f.write("{}  {}  {}  {}\n\n".format(
            C1.rjust(n1), C2.rjust(n2), C3.rjust(n3), C4.rjust(n4)))

        # url
        f.write("url = {}\n\n".format(url))
        f.write("=================================\n")

        # content
        f.write("X[:, :2], {}, {}, {}\n".format(
            str(type(X))[8:-2], X.shape, str(type(X[0, 0]))[8:-2]))
        if isinstance(X, sparse.spmatrix):
            f.write("{}\n".format(X[:show_n, :2].toarray()))
        else:
            f.write("{}\n".format(X[:show_n, :2]))
        f.write("...\n\n")

        f.write("y_true, {}, {}, {}\n".format(
            str(type(y_true))[8:-2], y_true.shape, str(type(y_true[0]))[8:-2]))
        f.write("{}".format(y_true[:show_n]))
        f.write("...\n\n")

        f.write("distribution\n")
        f.write(ind_L[:50].to_string())
        f.write("\n\n")

def WHH(W, c, beta=0.5, ITER=100):
    val, vec = np.linalg.eigh(W)
    H = vec[:, -c:]
    #  H = sparse.linalg.eigsh(W, which='LA', k=c)[1]
    #  print(np.mean(H))
    H = np.maximum(W @ H, 0.00001)

    obj = np.zeros(ITER)
    obj[0] = np.linalg.norm(W - H@H.T, ord="fro")

    for i in range(1, ITER):
        H_old = H.copy()

        WH = W@H
        HHH = H@(H.T@H)
        H = H*(1 - beta + beta*WH/HHH)

        obj[i] = np.linalg.norm(W - H@H.T, ord="fro")

        if np.abs(obj[i] - obj[i-1])/obj[i] < 1e-6:
            break

    return H, obj


#  def WHH2(W, lam, P, c, beta=0.5, ITER=100):
#      W = sparse.csr_matrix(W)
#      H = sparse.linalg.eigsh(W, which='LA', k=c)[1]
#      H = np.maximum(W @ H, 0.00001)
#
#      obj = np.zeros(ITER)
#      obj[0] = np.linalg.norm(W + lam*P@P.T - H@H.T, ord="fro")
#
#      for i in range(1, ITER):
#
#          WH = W@H + lam*P@(P.T@H)
#          HHH = H@(H.T@H)
#          H = H*(1 - beta + beta*WH/HHH)
#
#          obj[i] = np.linalg.norm(W + lam*P@P.T - H@H.T, ord="fro")
#
#          if np.abs(obj[i] - obj[i-1])/obj[i] < 1e-6:
#              break
#
#      return H, obj

def norm_W(W):
    d = np.sum(W, 1)
    d_inv = 1 / np.sqrt(d)
    W2 = W*np.outer(d_inv, d_inv)
    return W2



def rand_ring(r, N, tur, cx=0, cy=0, s=0, e=2 * np.pi, dis="gaussian"):
    """
    dis=["uniform", "gaussian"]
    """
    theta = np.linspace(s, e, N)
    x = np.vstack((r * np.cos(theta), r * np.sin(theta))).T
    x = x + turdata(N, tur, d=2, dis=dis)

    x[:, 0] += cx
    x[:, 1] += cy

    return x

def turdata(N, tur, d=2, dis="gaussian"):
    """
    dis=["uniform", "gaussian"]
    """

    if dis == "gaussian":
        mu = np.repeat(0, d)
        sig = np.eye(d) * tur
        x = np.random.multivariate_normal(mu, sig, N)
    elif dis == "uniform":
        x = np.random.uniform(-tur/2, tur/2, (N, 2))
    return x
