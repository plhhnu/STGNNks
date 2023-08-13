import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, \
                            homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scanpy as sc
import stlearn as st
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import sys
import matplotlib.pyplot as plt

import random
import sys
import sklearn.metrics as metrics
from s_dbw import S_Dbw
import os
import eval
import datetime
now1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("时间1:", now1)
# BASE_PATH = Path('../data/DLPFC')
#
# sample = sys.argv[1]
#
# TILE_PATH = Path("/tmp/{}_tiles".format(sample))
# TILE_PATH.mkdir(parents=True, exist_ok=True)
#
# OUTPUT_PATH = Path(f"../output/DLPFC/{sample}/stLearn")
# OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


#
# ground_truth_df = pd.read_csv( BASE_PATH / sample / 'metadata.tsv', sep='\t')
# ground_truth_df['ground_truth'] = ground_truth_df['layer_guess']
#
# le = LabelEncoder()
# ground_truth_le = le.fit_transform(list(ground_truth_df["ground_truth"].values))
#
# n_cluster = len((set(ground_truth_df["ground_truth"]))) - 1
# data.obs['ground_truth'] = ground_truth_df["ground_truth"]
#
# ground_truth_df["ground_truth_le"] = ground_truth_le
TILE_PATH = Path("/data/pxh/test/tiles")
TILE_PATH.mkdir(parents=True, exist_ok=True)

# proj_list = ['Ductal_Carcinoma_in_situ',
#              'Adult Mouse Brain (FFPE)','V1_Adult_Mouse_Brain_Coronal_Section_1',
#              'Adult Mouse Kidney (FFPE)','V1_Breast_Cancer_Block_A_Section_2']
proj_list = ['Ductal_Carcinoma_in_situ']
for proj_idx in range(len(proj_list)):
    #category = "Visium_Mouse_Olfactory_Bulb1.3"
    category =proj_list[proj_idx]
    director = f"/data/pxh/test/data/{category}"
    data = st.Read10X(f"{director}")
    #data = sc.read_visium(f"{director}")
    #data.var_names_make_unique()
    # pre-processing for gene count table
    st.pp.filter_genes(data,min_cells=1)
    st.pp.normalize_total(data)
    st.pp.log1p(data)

    # run PCA for gene expression data
    st.em.run_pca(data,n_comps=15)

    # pre-processing for spot image
    st.pp.tiling(data, TILE_PATH)

    # this step uses deep learning model to extract high-level features from tile images
    # may need few minutes to be completed
    st.pp.extract_feature(data)

    # stSME
    st.em.run_pca(data,n_comps=50)
    data_ = data.copy()
    st.spatial.SME.SME_normalize(data_, use_data="raw")

    data_.X = data_.obsm['raw_SME_normalized']
    st.pp.scale(data_)
    st.em.run_pca(data_,n_comps=50)
    st.pp.neighbors(data_)
    st.em.run_umap(data_)

    if category in ['V1_Breast_Cancer_Block_A_Section_2', 'V1_Breast_Cancer_Block_A_Section_1',
                    'V1_Adult_Mouse_Brain_Coronal_Section_1','Adult Mouse Brain (FFPE)']:
        n_clusters = 20
    else:
        n_clusters = 12


    def res_search_fixed_clus(adata, fixed_clus_count, increment=0.002):
        '''
            arg1(adata)[AnnData matrix]
            arg2(fixed_clus_count)[int]

            return:
                resolution[int]
        '''
        for res in sorted(list(np.arange(0.4, 2.5, increment)), reverse=True):
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique_leiden = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            if count_unique_leiden == fixed_clus_count:
                break
        return res


    eval_resolution = res_search_fixed_clus(data_, n_clusters)
    st.tl.clustering.louvain(data_, resolution=eval_resolution)
    #st.tl.clustering.kmeans(data_, n_clusters=n_clusters, use_data="X_pca", key_added="X_pca_kmeans")
    st.pl.cluster_plot(data_, use_label="louvain")


    X = data_.obsm['X_pca']
    y = data_.obs['louvain']
    y = y.values.reshape(-1)
    y = y.codes
    n_cluster = len(np.unique(np.array(y)))

    parameter = f"cluster{n_clusters},resolution{eval_resolution}"
    if not os.path.exists(f'./outputs/{category}'):
        os.makedirs(f'./outputs/{category}')
    # # read the annotation
    # Ann_df = pd.read_csv(f'{director}/metadata.tsv', sep='\t')
    # ARI = np.round(metrics.adjusted_rand_score(y, Ann_df['fine_annot_type']), 4)
    # NMI = np.round(metrics.normalized_mutual_info_score(y, Ann_df['fine_annot_type']), 4)
    # import csv
    #
    # with open(f'./outputs/{category}/ARI_NMI.csv', mode='a+') as f:
    #     f_writer = csv.writer(
    #         f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #     f_writer.writerow([str("  ")])
    #     f_writer.writerow([str(now)])
    #     f_writer.writerow(["ARI_list", str(ARI)])
    #     f_writer.writerow(["NMI_list", str(NMI)])
    #     f_writer.writerow(["parameter", str(parameter)])

    if not os.path.exists(f'./outputs/{category}'):
        os.makedirs(f'./outputs/{category}')

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.spatial(data_, img_key="hires", color="louvain", size=1.5, show=False,title='stLearn')
    plt.savefig(f'./outputs/{category}/atest.jpg',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'./outputs/{category}/size=1.5_{parameter}.jpg',
                bbox_inches='tight', dpi=300)
    sc.pl.spatial(data_, img_key="hires", color="louvain", size=1.2, show=False,title='stLearn')
    plt.savefig(f'./outputs/{category}/size=1.2_{parameter}.jpg',
                bbox_inches='tight', dpi=300)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.umap(data_, color='louvain', legend_loc='on data', s=20, legend_fontoutline='2', show=False,title='stLearn')
    plt.savefig(f'./outputs/{category}/umap_{parameter}.jpg',
                bbox_inches='tight', dpi=300)
    # plt.rcParams["figure.figsize"] = (5, 5)
    # sc.pl.umap(data_, color=['X_pca_kmeans'], legend_loc='on data', s=10, show=False)



    dav = np.round(metrics.davies_bouldin_score(X, y), 5)
    cal = np.round(metrics.calinski_harabasz_score(X, y), 5)
    sil = np.round(metrics.silhouette_score(X, y), 5)
    # table = Counter(y)
    table = []
    sdbw = np.round(S_Dbw(X, y), 5)

    eval.Spatialeval(os.path.join(f"./outputs/{category}/", f"{category}_index.csv"),
                     X, y, X.shape[1], dav, cal, sil, sdbw, table, parameter)

    print(data_.isbacked)
    if not os.path.exists(f'./h5ad/{category}'):
        os.makedirs(f'./h5ad/{category}')
    data_.filename = f'./h5ad/{category}/final_{category}_{n_cluster}.h5ad'
    print(data_.isbacked)
    now2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("时间2:", now2)
    print("dav:", dav)
    print("cal:", cal)
    print("sil:", sil)
    print("sdbw:", sdbw)

print("finish")
