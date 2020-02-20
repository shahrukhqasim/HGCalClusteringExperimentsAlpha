# HGCal Clustering Experiments
This repository contains inital clustering experiments on HGCal data based on ragged tensors and object condensation loss

Setup the HGCal ML repo:
```
https://github.com/jkiesele/HGCalML
```

Ask @jkiesele for which singularity container to use which should be easiest to set up.

Then you should be able to run training using:

```
python training.py
```

It's not very user friendly as of now since I have to do more tests in eager mode. I am working on keras version (comning soon). 
