## Overview
<p align="center">
  <img width="700"  src="https://github.com/STGNNks-master/framework/framework.jpg">
</p>

## Data
Data is available at (https://www.10xgenomics.com/resources/datasets)

## Environment
Install python3 for running this code. And these packages should be satisfied:
* tensorflow == 1.1.0
* pandas == 1.1.5
* numpy == 1.19.5
* stlrarn==0.3.2
* matplotlib == 3.2.2
* pytorch==1.10.2
* sklearn==0.0

## install
```
python setup.py build_ext --inplace
```

## Usage
the first step, to run data_generation_ST.py to preprocess the raw 10x ST data.

the second step, to run the run_STGNNks.py to load the 10x ST data.

the final step, to run the run_STGNNks.py to call the algorithm model, the model is in ST_KSUM.py .the method are detailed in the paper.
```
python ST_KSUM.py

```
