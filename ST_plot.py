import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
import pandas as pd
import scanpy as sc
import stlearn as st


adata = sc.read_visium("./Adult Mouse Brain (FFPE)")
X_embedding = pd.read_csv(f'./embedding matrix.csv')
X_embedding = X_embedding.values[:, 1:]
adata.var_names_make_unique()
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
sc.pp.log1p(adata)
print(pd.DataFrame(X_embedding))
cell_reps = pd.DataFrame(X_embedding)
cell_reps.index = adata.obs.index
adata.obsm["ST"] = cell_reps.loc[adata.obs_names,].values
sc.pp.neighbors(adata, use_rep='ST')
sc.tl.umap(adata)

print("-----------Data Visualization-------------")
y_preds = pd.read_csv(f"./knn.csv", header=0, index_col=0)#KNN根据不同参数使用不同csv文件名
y_pred = y_preds.iloc[:, 0:1]
y_pred = y_pred.values.reshape(-1)
adata.obs['clusters'] = pd.Categorical(
            values=y_pred.astype('U'),
            categories=natsorted(map(str, np.unique(y_pred))),
        )
plt.rcParams["figure.figsize"] = (4, 4)
sc.pl.umap(adata, color=['clusters'], legend_loc='on data', s=30, show=False,title='STGNNKs')
plt.savefig(f"./Umap.jpg")
plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(adata, img_key="hires", color="clusters", size=1.5,title='STGNNKs')
plt.savefig(f'./clusters.jpg', bbox_inches='tight',dpi=150)
#热图
sc.tl.rank_genes_groups(adata, "clusters", method="t-test")
sc.pl.rank_genes_groups_heatmap(adata,n_genes=5, groupby="clusters")
plt.savefig(f'./clusters_heatmap.jpg')
# 可视化差异基因
sc.pl.rank_genes_groups(adata, n_genes=5, sharey=False)
plt.savefig(f'./clusters_genes.jpg')
sc.pl.rank_genes_groups_dotplot(adata, n_genes=4)
plt.savefig(f'./rank_genes_groups_dotplot.jpg')
# 输出每一簇
pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5)
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']}).head(5)
res = pd.DataFrame({group + '_' + key: result[key][group] for group in groups for key in
                                ['names', 'pvals', 'logfoldchanges', 'pvals_adj', 'scores']})
res.to_csv(f"./sdif_markers.csv")



