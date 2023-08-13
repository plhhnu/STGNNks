import sklearn.metrics as metrics
import csv
import datetime
import numpy as np
from s_dbw import S_Dbw
import scanpy as sc
import pandas as pd

def DLPFCeval(dataroot, dataname, y):
    # add ground_truth
    adata = sc.read_visium(os.path.join(dataroot, dataname))
    adata.var_names_make_unique()
    
    df_meta = pd.read_csv(dataroot + f'/{dataname}/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs['ground_truth'] = df_meta_layer.values
    adata.obs['seurat'] = y

    # filter out NA nodes
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    ARI = np.round(metrics.adjusted_rand_score(adata.obs['seurat'], adata.obs['ground_truth'] ), 5)
    NMI = np.round(metrics.normalized_mutual_info_score(adata.obs['seurat'], adata.obs['ground_truth']), 5)

    with open(f'./outputs-DLPFC/ARI_NMI.csv', mode='a+') as f:
        f_writer = csv.writer(
                f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        f_writer.writerow([str(now)])
        f_writer.writerow([str(dataname),"ARI:" + str(ARI), "NMI:" + str(NMI)])


def title(output_CSV):
    
    with open(output_CSV, mode='a+') as f:
        f_writer = csv.writer(
            f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow([" "])   
        
def seurateval(output_CSV,X,y,npcs,num,n):
    dav = np.round(metrics.davies_bouldin_score(X, y), 5)
    cal = np.round(metrics.calinski_harabasz_score(X, y), 5)
    sil = np.round(metrics.silhouette_score(X, y), 5)
    sdbw = np.round(S_Dbw(X, y), 5)
    k = len(np.unique(np.array(y)))

    now = datetime.datetime.now()
    with open(output_CSV, mode='a+') as f:
        f_writer = csv.writer(
            f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow([str(npcs), str(num),str(k),str(n), 
        str(dav), str(cal), str(sil),str(sdbw),str(now)])
        # f_writer = csv.writer(
        #     f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # f_writer.writerow(['brain', 'k=',str(k),' ',str(now)])
        # f_writer.writerow(['','pc.num=1:'+str(num)])
        # f_writer.writerow(['dav= ' + str(dav), ' ', 'cal= ' + str(cal), ' ', 'sil= ' + str(sil)])
        # f_writer.writerow(['--------------------------'])

