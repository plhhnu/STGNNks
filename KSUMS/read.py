import scanpy as sc
adata = sc.read("G:\Single-cell\GSE147729_RAW\GSM4443875_young_A_matrix.mtx\young_A_matrix.mtx")
data = adata.X
data = data.todense()
print(data)