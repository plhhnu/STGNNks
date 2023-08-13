import numpy as np
import anndata
import scanpy as sc
from natsort import natsorted
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn.metrics as metrics

def plot(x,y_pred,num,output_dir,file):
    #DLPFC
    path = f"E:/DLPFC/{file}"
    #path = f"./data/{file}"
    adata = sc.read_visium(path)
    num = num
    k = len(np.unique(np.array(y_pred)))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)
    sc.pp.pca(adata,n_comps=50)
    adata.obsm['X_pca'] = x
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added="Seurat")
    y_pred = np.array(y_pred)
    adata.obs['Seurat'] = pd.Categorical(
    	values=y_pred.astype('U'),
    	categories=natsorted(map(str, np.unique(y_pred))),
    )
    
    # #breast1
    # Ann_df = pd.read_csv('E:/V1_Breast_Cancer_Block_A_Section_1/metadata.tsv', sep='\t')
    # ARI = np.round(metrics.adjusted_rand_score(adata.obs['Seurat'], Ann_df['fine_annot_type']), 5)
    # NMI = np.round(metrics.normalized_mutual_info_score(adata.obs['Seurat'], Ann_df['fine_annot_type']), 5)
    # now = datetime.datetime.now()
    # import csv
    # with open(f"{output_dir}/Seurat-breast1-SCT3000.csv", mode='a+') as f:
    #     f_writer = csv.writer(
    #         f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #     f_writer.writerow([str("  ")])
    #     f_writer.writerow([str(now)])
    #     f_writer.writerow(["ARI",str(ARI)])
    #     f_writer.writerow(["NMI",str(NMI)])
    plt.rcParams["figure.figsize"] = (3, 3)
    if not os.path.exists(f'{output_dir}/{file}'):
        os.makedirs(f'{output_dir}/{file}')
    sc.pl.umap(adata, color=["Seurat"], legend_loc='on data',s=20,legend_fontoutline='2',show=False,title='Seurat')
    plt.savefig(f'{output_dir}/{file}/umap,seurat,.jpg', bbox_inches='tight', dpi=300)
    sc.pl.spatial(adata, img_key="hires", color=['Seurat'], show=False,size = 1.2,title='Seurat')
    plt.savefig(f'{output_dir}/{file}/size = 1.2,seurat,{file}.jpg', bbox_inches='tight', dpi=300)
    sc.pl.spatial(adata, img_key="hires", color=['Seurat'], show=False,size = 1.5,title='Seurat')
    plt.savefig(f'{output_dir}/{file}/size = 1.5,seurat,{file}.jpg', bbox_inches='tight', dpi=300)




