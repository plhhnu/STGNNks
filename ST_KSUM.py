##exocrine GCNG with normalized graph matrix
import os
import sys
import matplotlib

matplotlib.use('Agg')
# matplotlib.use('TkAgg')
import anndata as ad
from STGNNks import get_graph, train_DGI, train_DGI
from datetime import datetime
import funs as Ifuns
from keras.layers import Dense, Input, GaussianNoise, Layer, Activation
import keras.backend as K
from keras.models import Model
from keras.engine.topology import Layer, InputSpec
from keras.optimizers import SGD, Adam
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from KSUMS import KSUMS
from layers import ConstantDispersionLayer, SliceLayer, ColWiseMultLayer
from loss import poisson_loss, NB, ZINB
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from natsort import natsorted
from numpy.random import seed
import pylab as pl
import pandas as pd
import pickle
from preprocess import read_dataset, normalize
import random
from s_dbw import S_Dbw
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import SpatialDE
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import sparse
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool  # noqa
from torch_geometric.data import Data, DataLoader
import tensorflow as tf
from time import time
import timeit
from tensorflow import set_random_seed
# import os
# os.environ["CUDA_LAUNCH_BLOCKING"]= "1"
seed = 2
random.seed(seed)

MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)


def get_data(args):

    data_file = args.data_path + args.data_name + '/'
    with open(data_file + 'Adjacent', 'rb') as fp:
        adj_0 = pickle.load(fp)

    X_data = np.load(data_file + 'features.npy')

    num_points = X_data.shape[0]
    adj_I = np.eye(num_points)
    adj_I = sparse.csr_matrix(adj_I)
    adj = (1 - args.lambda_I) * adj_0 + args.lambda_I * adj_I
    return adj_0, adj, X_data



def processing(adata, size_factors=True):#, normalize_input=True, logtrans_input=True):
    if size_factors:# or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        # normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    return adata

def autoencoder(dims, noise_sd=0, init='glorot_uniform', act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        Model of autoencoder
    """
    n_stacks = len(dims) - 1
    # input
    sf_layer = Input(shape=(1,), name='size_factors')
    x = Input(shape=(dims[0],), name='counts')
    h = x
    h = GaussianNoise(noise_sd, name='input_noise')(h)

    # internal layers in encoder
    for i in range(n_stacks - 1):
        h = Dense(dims[i + 1], kernel_initializer=init, name='encoder_%d' % i)(h)
        h = GaussianNoise(noise_sd, name='noise_%d' % i)(h)  # add Gaussian noise
        h = Activation(act)(h)

    h = Dense(dims[-1], kernel_initializer=init, name='encoder_hidden')(h)  # hidden layer, features are extracted from here

   # internal layers in decoder
    for i in range(n_stacks - 1, 0, -1):
        h = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(h)

    # output

    pi = Dense(dims[0], activation='sigmoid', kernel_initializer=init, name='pi')(h)

    disp = Dense(dims[0], activation=DispAct, kernel_initializer=init, name='dispersion')(h)

    mean = Dense(dims[0], activation=MeanAct, kernel_initializer=init, name='mean')(h)

    output = ColWiseMultLayer(name='output')([mean, sf_layer])
    output = SliceLayer(0, name='slice')([output, disp, pi])

    return Model(inputs=[x, sf_layer], outputs=output)

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class model(object):

    def __init__(self,
                 n_clusters,
                 dims,
                 noise_sd=2.5,
                 alpha=1.0,
                 ridge=0,
                 debug=False):

        super(model, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.noise_sd = noise_sd
        self.alpha = alpha
        self.act = 'relu'
        self.ridge = ridge
        self.debug = debug
        self.autoencoder = autoencoder(self.dims, noise_sd=self.noise_sd, act=self.act)
        print("autoencoder:", self.autoencoder)
        # prepare clean encode model without Gaussian noise
        ae_layers = [l for l in self.autoencoder.layers]
        hidden = self.autoencoder.input[0]
        for i in range(1, len(ae_layers)):
            if "noise" in ae_layers[i].name:
                next
            elif "dropout" in ae_layers[i].name:
                next
            else:
                hidden = ae_layers[i](hidden)
            if "encoder_hidden" in ae_layers[i].name:  # only get encoder layers
                break
        self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)

        pi = self.autoencoder.get_layer(name='pi').output
        disp = self.autoencoder.get_layer(name='dispersion').output
        mean = self.autoencoder.get_layer(name='mean').output
        zinb = ZINB(pi, theta=disp, ridge_lambda=self.ridge, debug=self.debug)
        self.loss = zinb.loss
        clustering_layer = ClusteringLayer(self.n_clusters, alpha=self.alpha, name='clustering')(hidden)
        self.model = Model(inputs=[self.autoencoder.input[0], self.autoencoder.input[1]],
                           outputs=[clustering_layer, self.autoencoder.output])


    def pretrain(self, x, y,
        batch_size = 800,
        epochs = 10,
        optimizer = 'adam',
        ae_file = None):
        print('...Pretraining autoencoder...')
        self.autoencoder.compile(loss=self.loss, optimizer=optimizer)
        es = EarlyStopping(monitor="loss", patience=50, verbose=1)
        self.autoencoder.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, callbacks=[es])
        self.autoencoder.save_weights(ae_file )
        print('Pretrained weights are saved to ./' + str(ae_file))
        self.pretrained = True

    def fit(self, x_counts, sf,
            batch_size=256,
            ae_weights=None,
            loss_weights=[1,1],
            optimizer='adadelta'):
        self.model.compile(loss=['kld', self.loss], loss_weights=loss_weights, optimizer=optimizer)
        save_interval = int(x_counts.shape[0] / batch_size) * 5  # 5 epochs
        print('Save interval', save_interval)

        if not self.pretrained and ae_weights is None:
            print('...pretraining autoencoders using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=200')
            self.pretrain(x_counts, batch_size)
            self.pretrained = True
        elif ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print('ae_weights is loaded successfully.')
        ada = self.encoder.predict([x_counts, sf])

        return ada


def STGNNks_on_ST(args):
    lambda_I = args.lambda_I
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj_0, adj, X_data = get_data(args)  # , cell_type_indeces

    num_cell = X_data.shape[0]
    num_feature = X_data.shape[1]
    print('Adj:', adj.shape, 'Edges:', len(adj.data))
    print('X:', X_data.shape)



    if args.DGI and (lambda_I >= 0):
        print("-----------Deep Graph Infomax-------------")
        data_list = get_graph(adj, X_data)
        data_loader = DataLoader(data_list, batch_size=batch_size)
        DGI_model = train_DGI(args, data_loader=data_loader, in_channels=num_feature)

        for data in data_loader:
            data.to(device)
            X_embedding, _, _ = DGI_model(data)
            X_embedding = X_embedding.cpu().detach().numpy()
            X_embedding_filename = args.embedding_data_path + 'lambdaI' + str(lambda_I) + '_epoch' + str(
                args.num_epoch) + '_Embed_X.npy'
            np.save(X_embedding_filename, X_embedding)



        print("-----------Dimension reduction-------------")

        X_embedding_filename = args.embedding_data_path + 'lambdaI' + str(lambda_I) + '_epoch' + str(
            args.num_epoch) + '_Embed_X.npy'
        X_embedding = np.load(X_embedding_filename)
        datas = sc.AnnData(X_embedding)
        datas = processing(datas,
                           size_factors=True,
                           )
        input_size = datas.n_vars
        result_fp = open("./result_matric.csv", mode="w")
        c_true=10
        pred_list = []
        Encode = model(dims=[input_size, 200, 100,20], n_clusters=c_true, noise_sd=2.5)
        y = datas.raw.X
        Encode.pretrain(x=[datas.X, datas.obs.size_factors], y=y, batch_size=args.batch_size,
                                epochs=args.pretrain_epochs, optimizer=Adam(amsgrad=True),
                                ae_file=args.ae_weight_file)

        X_embedding = Encode.fit(x_counts=datas.X, sf=datas.obs.size_factors, batch_size=args.batch_size,
                                         ae_weights=args.ae_weights, loss_weights=[args.gamma, 1], optimizer='adadelta')

        df = pd.DataFrame(X_embedding)
        df = df.fillna(0)

        print("-----------Clustering-------------")

        # result_fp.write("knn_{},c_true, dav, cal, sil,sdbw\n".format(knn))
        knn=500

        result_fp.write("class_num_{},".format(c_true))
        D = Ifuns.EuDist2(X_embedding, X_embedding, squared=True)
        np.fill_diagonal(D, -1)
        ind_M = np.argsort(D, axis=1)
        np.fill_diagonal(D, 0)
        NN = ind_M[:, :knn]
        NND = Ifuns.matrix_index_take(D, NN)
        obj = KSUMS(NN.astype(np.int32), NND.astype(np.double), c_true)
        obj.clu()
        y_pred = obj.y_pre
        pred_list.append(np.array(y_pred).reshape(-1, 1))
        dav = metrics.davies_bouldin_score(X_embedding, y_pred)
        cal = metrics.calinski_harabasz_score(X_embedding, y_pred)
        sil = metrics.silhouette_score(X_embedding, y_pred)
        sdbw = np.round(S_Dbw(X_embedding, y_pred), 5)
        # Ann_df = pd.read_csv('./metadata.tsv', sep='\t')
        # ARI = np.round(metrics.adjusted_rand_score(y_pred, Ann_df['fine_annot_type']), 3)
        # NMI = np.round(metrics.normalized_mutual_info_score(y_pred, Ann_df['fine_annot_type']), 3)


        all_data = []  # txt: cell_id, cell batch, cluster type

        for index in range(num_cell):
            all_data.append([index, y_pred[index]])
        np.savetxt(f"‘./types.txt", np.array(all_data),fmt='%3d', delimiter='\t')
        pd.DataFrame(np.hstack(pred_list))







