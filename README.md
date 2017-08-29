# Examples for Universal Training Control Tools

A set examples of using UTCT for various classification/regression tasks.

## Remarks
All scripts must be runned from the root of the repository, e.g. for MXNet-version of MNIST training:
```
python mnist/MXNet/train.py
```

## Visualization of learning curves
Any training script usually creates a project subdirectory in the working director with all 
working files. One of them is a `*_score.log` file with tabulated learning curves. You can visualize 
this curves by using R-script `utct/common/observe.R`. This script starts a web-server and has two 
arguments:
- the path to `score_ref.csv` file (this file is in the working directory),
- port for web-server (the defalut value is 6006).
After starting this script you can see curve plots in your web-browser by the address 
`http://localhost:6006`.