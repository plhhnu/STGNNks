import stlearn as st
from pathlib import Path
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import numpy as  np
import matplotlib as mpl
import pandas as pd
from natsort import natsorted
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
st.settings.set_figure_params(dpi=120)
# Reading data
data = st.Read10X(path="./V1_Breast_Cancer_Block_A_Section_1/")
X_embedding=pd.read_csv(f'./embedding matrix_20.csv')
X_embedding=X_embedding.values[:,1:]
data.layers["raw_count"] = data.X
# Preprocessing
st.pp.filter_genes(data,min_cells=3)
st.pp.normalize_total(data)
st.pp.log1p(data)
# Keep raw data
data.raw = data
st.pp.scale(data)

# Run PCA
st.em.run_pca(data,n_comps=50,random_state=0)
# Tiling image
st.pp.tiling(data,out_path="tiling",crop_size = 40)
# Using Deep Learning to extract feature
st.pp.extract_feature(data)
# Apply stSME spatial-PCA option
data.obsm["ST_embedding"]=X_embedding
st.spatial.morphology.adjust(data,use_data="ST_embedding",radius=50,method="mean")
cell_reps = pd.DataFrame(X_embedding)
cell_reps.index=data.obs.index
data.obsm["ST"] = cell_reps.loc[data.obs_names,].values
st.pp.neighbors(data,n_neighbors=25,use_rep='ST',random_state=0)#KNN根据不同参数使用不同csv文件名
st.tl.clustering.louvain(data,random_state=0)
sc.tl.umap(data)
y_pred = pd.read_csv(f"./knn.csv", header=0, index_col=0)
y_pred = y_pred.iloc[:, 0:1]
y_pred = y_pred.values.reshape(-1)
print(y_pred)
data.obs['clusters'] = pd.Categorical(
         values=y_pred.astype('U'),
         categories=natsorted(map(str, np.unique(y_pred))),
       )
data.uns["iroot"] = st.spatial.trajectory.set_root(data,use_label="clusters",cluster=3,use_raw=True)
st.spatial.trajectory.pseudotime(data,eps=50,use_rep="X_pca",use_label="clusters")

sc.tl.rank_genes_groups(data, "clusters", method="t-test")
st.spatial.trajectory.pseudotimespace_global(data,use_label="clusters",list_clusters=["4","1"])
st.pl.cluster_plot(data,use_label="clusters",show_trajectories=True,list_clusters=["4","1"],show_subcluster=True)
plt.savefig(f"./pseudotimespace_globa_cluster_plot.jpg")

st.pl.trajectory.tree_plot(data,use_label="clusters")
plt.savefig(f"./tree_plot.jpg")

st.spatial.trajectory.detect_transition_markers_clades(data,clade=23,use_raw_count=True)
st.pl.trajectory.transition_markers_plot(data,top_genes=10,trajectory="clade_23")
plt.savefig(f"./transition_markers_plot.jpg")


