import warnings
warnings.filterwarnings("ignore")
import datetime
now1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("时间1:", now1)
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import random
import sys
import sklearn.metrics as metrics
from s_dbw import S_Dbw
from collections import Counter
from natsort import natsorted
import STAGATE
import os
import eval
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

random.seed(1234)
np.random.seed(1234)

proj_list = ['Adult Mouse Brain (FFPE)']
for proj_idx in range(len(proj_list)):
    category = proj_list[proj_idx]
    director = f"/data/pxh/SEDR/data/{category}"
    adata = sc.read_visium(f"{director}")
    adata.var_names_make_unique()

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if not os.path.exists(f'./outputs/{category}'):
        os.makedirs(f'./outputs/{category}')
    sc.pp.pca(adata, n_comps=30)
    sc.pp.neighbors(adata)

    def res_search_fixed_clus(adata, fixed_clus_count, increment=0.01):
        '''
            arg1(adata)[AnnData matrix]
            arg2(fixed_clus_count)[int]

            return:
                resolution[int]
        '''
        for res in sorted(list(np.arange(0.3, 3, increment)), reverse=True):
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique_louvain = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            if count_unique_louvain == fixed_clus_count:
                break
        return res

    # eval_resolution = res_search_fixed_clus(adata, 12)
    eval_resolution = 1
    sc.tl.louvain(adata, resolution=eval_resolution)

    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.umap(adata, color=['louvain'], legend_loc='on data', s=20,show=False)
    plt.savefig(f'./outputs/{category}/umap_scanpy.jpg',
                bbox_inches='tight', dpi=150)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.spatial(adata, img_key="hires", color="mclust", legend_loc='right margin',
                  size=1.5, show=False)
    plt.savefig(f'./outputs/{category}/domain{ARI}.jpg',
                bbox_inches='tight', dpi=300)

    adata.uns['louvain_colors']=['#aec7e8', '#9edae5', '#d62728', '#dbdb8d', '#ff9896',
                                 '#8c564b', '#696969', '#778899', '#17becf', '#ffbb78',
                                 '#e377c2', '#98df8a', '#aa40fc', '#c5b0d5', '#c49c94',
                                 '#f7b6d2', '#279e68', '#b5bd61', '#ad494a', '#8c6d31',
                                 '#1f77b4', '#ff7f0e']

    #read the annotation
    Ann_df = pd.read_csv(f'{director}/metadata.tsv', sep='\t')


    ARI = np.round(metrics.adjusted_rand_score(adata.obs['louvain'], Ann_df['fine_annot_type']), 4)
    NMI = np.round(metrics.normalized_mutual_info_score(adata.obs['louvain'], Ann_df['fine_annot_type']), 4)
    import csv

    with open(f'./outputs/{category}/ARI_NMI.csv', mode='a+') as f:
        f_writer = csv.writer(
            f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f_writer.writerow([str(now)])
        f_writer.writerow(["ARI_list",str(ARI)])
        f_writer.writerow(["NMI_list",str(NMI)])
        #f_writer.writerow([f"cluster{n_cluster}", str(parameter)])
    X = adata.obsm['X_pca']
    y = adata.obs['louvain']
    y = y.values.reshape(-1)
    y = y.codes
    n_cluster = len(np.unique(np.array(y)))

    dav = np.round(metrics.davies_bouldin_score(X, y), 5)
    cal = np.round(metrics.calinski_harabasz_score(X, y), 5)
    sil = np.round(metrics.silhouette_score(X, y), 5)
    sdbw = np.round(S_Dbw(X, y), 5)
    table = []
    parameter = f"cluster{n_cluster},eval_resolution{eval_resolution}"

    plt.rcParams["figure.figsize"] = (5, 5)
    sc.pl.spatial(adata, img_key="hires", color="louvain", size=1.2,show=False,title='SCANPY')
    plt.savefig(f'./outputs/{category}/size=1.2,{parameter}.jpg',
                bbox_inches='tight', dpi=150)
    sc.pl.spatial(adata, img_key="hires", color="louvain", size=1.5,show=False,title='SCANPY')
    plt.savefig(f'./outputs/{category}/size=1.5,{parameter}.jpg',
                bbox_inches='tight', dpi=150)

    plt.rcParams["figure.figsize"] = (5, 5)
    sc.pl.umap(adata, color=['louvain'], legend_loc='on data', s=10,show=False,title='SCANPY')
    plt.savefig(f'./outputs/{category}/umap_scanpy_{parameter}.jpg',
                bbox_inches='tight', dpi=150)

    eval.Spatialeval(os.path.join(f"./outputs/{category}/", f"{category}_louvain_index.csv"),
                         X, y, X.shape[1], dav, cal, sil, sdbw, table,parameter)
    print(adata.isbacked)
    if not os.path.exists(f'./h5ad/{category}'):
        os.makedirs(f'./h5ad/{category}')
    adata.filename = f'./h5ad/{category}/final_scanpy_{parameter}.h5ad'
    # print(adata.isbacked)
    now2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("时间2:", now2)
