library(tidyverse)
library(Seurat)
library(SeuratData)
library(ggplot2)
library(patchwork)
library(dplyr)
library(reticulate)

Brain <- Load10X_Spatial('./data/Adult Mouse Brain (FFPE)')
#Brain <- Load10X_Spatial('./data/Adult Mouse Brain (Coronal)')
Brain <- SCTransform(Brain, assay = "Spatial", variable.features.n = 3000, verbose = FALSE)

ScaleData <- Brain@assays$SCT@scale.data
npcs=50
Brain1 = Brain
Brain = Brain1
num = 30
n_clusters = 20
source_python('seurat-eval.py')
output_CSV = "./outputs-brain/Seurat-brain-SCT3000.csv"
output_CSV = "./Adult Mouse Brain (FFPE)/Adult Mouse Brain (FFPE)-SCT3000.csv"
title(output_CSV)
while(n_clusters<21){
  Brain = Brain1
  Brain <- RunPCA(Brain, assay = "SCT",npcs = npcs, verbose = FALSE)
  pc.num = 1:num
  Brain <- FindNeighbors(Brain, reduction = "pca", dims = pc.num)
  # # selection of clusters
  # for(resolution in 300:1){
  #   Brain <- FindClusters(Brain, verbose = F, resolution = resolution/100)
  #   if(length(levels(Brain@meta.data$seurat_clusters)) == n_clusters){
  #     break
  #   }
  # }
  Brain <- FindClusters(Brain, verbose = FALSE)
  Brain <- RunUMAP(Brain, reduction = "pca", dims = pc.num)
  
  num = 30
  Brain <- FFPEbrain
  
  #
  label <- Brain@meta.data$seurat_clusters
  n = table(label)
  n
  test0 <- Brain@reductions$pca@cell.embeddings[,1:num]

  label <- as.integer(label)
  
  saveRDS(Brain, file="FFPEbrain.rds")
  
  seurateval(output_CSV,test0,label,npcs,num,n)
  #visualization
  source_python('plot.py')
  x = test0
  output_dir = "./outputs-brain"
  file = "Adult Mouse Brain (FFPE)"

  plot(test0,label,num,output_dir,file)
  n_clusters = n_clusters + 1
}


#p1 <- DimPlot(Brain, reduction = "umap", label = TRUE)
#p2 <- SpatialDimPlot(Brain, label = TRUE, label.size = 3)
#p1 + p2
