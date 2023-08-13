##exocrine GCNG with normalized graph matrix 
import os
import sys
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import pylab as pl
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sklearn import metrics
from scipy import sparse
#from sklearn.metrics import roc_curve, auc, roc_auc_score

import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool  # noqa
from torch_geometric.data import Data, DataLoader
from datetime import datetime 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # ================Specify data type firstly===============
    parser.add_argument( '--data_type', default='nsc', help='"nsc", \
        refers to non single cell resolution data(e.g. ST) respectively')
    # =========================== args ===============================
    parser.add_argument('--data_name', type=str, default='Adult Mouse Brain', help='The name of dataset')
    parser.add_argument( '--lambda_I', type=float, default=0.2)#0.3,0.4
    parser.add_argument( '--data_path', type=str, default='generated_data/', help='data path')
    parser.add_argument( '--model_path', type=str, default='model') 
    parser.add_argument( '--embedding_data_path', type=str, default='Embedding_data') 
    parser.add_argument( '--result_path', type=str, default='results') 
    parser.add_argument( '--DGI', type=int, default=1, help='run Deep Graph Infomax(DGI) model, otherwise direct load embeddings')
    parser.add_argument( '--load', type=int, default=0, help='Load pretrained DGI model')
    parser.add_argument( '--num_epoch', type=int, default=500, help='numebr of epoch in training DGI')
    parser.add_argument( '--hidden', type=int, default=256, help='hidden channels in DGI')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--pretrain_epochs', default=10, type=int)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--gamma', default=1, type=float,help='coefficient of clustering loss')
    parser.add_argument('--ae_weight_file', default='ae_weights.h5')
    args = parser.parse_args() 

    args.embedding_data_path = args.embedding_data_path +'/'+ args.data_name +'/'
    args.model_path = args.model_path +'/'+ args.data_name +'/'
    args.result_path = args.result_path +'/'+ args.data_name +'/'
    if not os.path.exists(args.embedding_data_path):
        os.makedirs(args.embedding_data_path) 
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path) 
    args.result_path = args.result_path+'lambdaI'+str(args.lambda_I) +'/'
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)


    print ('------------------------Model and Training Details--------------------------')
    print(args) 

    if args.data_type == 'nsc':
        from ST_KSUM import STGNNks_on_ST
        STGNNks_on_ST(args)
    else:
        print('Data type not specified')



    
