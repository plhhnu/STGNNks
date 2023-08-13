import pandas as pd
import numpy as np
path="D:/研一/KSUMS-master/data/dxb/feature.txt"


df = pd.read_csv(path, sep='\t',header=None,index_col=None)
df.to_csv("D:/研一/KSUMS-master/data/dxb/feature.csv",sep=",")
print(df)