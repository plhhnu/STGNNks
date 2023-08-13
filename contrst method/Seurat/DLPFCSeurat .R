library(tidyverse)
library(Seurat)
library(SeuratData)
library(ggplot2)
library(patchwork)
library(dplyr)
library(reticulate)
proj_list = ['151507', '151508', '151509', '151510',
              '151669', '151670', '151671', '151672',
              '151673', '151674', '151675', '151676']
if data_name in ['151669', '151670', '151671', '151672']:
    n_clusters = 5
else:
    n_clusters = 7

dataname <- "151676"
n_clusters = 7
dataroot <- "E:/DLPFC"
Brain <- Load10X_Spatial('E:/DLPFC/151676')

Brain <- SCTransform(Brain, assay = "Spatial", variable.features.n = 3000, verbose = FALSE)

ScaleData <- Brain@assays$SCT@scale.data
npcs=50
Brain1 = Brain
Brain = Brain1
num = 30

source_python('seurat-eval.py')
output_CSV = "./outputs-DLPFC/ARI_NMI.csv"

Brain = Brain1
Brain <- RunPCA(Brain, assay = "SCT",npcs = npcs, verbose = FALSE)

pc.num = 1:num
Brain <- FindNeighbors(Brain, reduction = "pca", dims = pc.num)
for(resolution in 200:1){
  Brain <- FindClusters(Brain, verbose = F, resolution = resolution/100)
  if(length(levels(Brain@meta.data$seurat_clusters)) == n_clusters){
    break
  }
}
#Brain <- FindClusters(Brain, verbose = FALSE)
Brain <- RunUMAP(Brain, reduction = "pca", dims = pc.num)

#
label <- Brain@meta.data$seurat_clusters
n = table(label)
n
test0 <- Brain@reductions$pca@cell.embeddings[,1:num]
label <- as.integer(label)
#saveRDS(Brain, file="151676.rds")

DLPFCeval(dataroot, dataname, label)
#可视化
source_python('plot.py')
x = test0
output_dir = "./outputs-DLPFC"
file = "151676"
plot(test0,label,num,output_dir,file)
#n_clusters = n_clusters + 1


#seurateval(test0,cidrlabel,50)


# p1 <- DimPlot(Brain, reduction = "umap", label = TRUE)
# p2 <- SpatialDimPlot(Brain, label = TRUE, label.size = 3)
# p1 + p2
