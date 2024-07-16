# Learning Parameters of Black-Box Models Using Differentiable Surrogates

This repository contains the scripts needs to recreate the results mentioned in the submission "Learning Parameters of Black-Box Models Using Differentiable Surrogates"


## Setup
Create a python virtual environment and install the required packages using
```bash
pip install -r requirements.txt
```

We have used Anaconda package manager in the Linux environment (Ubuntu 22.04.3 LTS). Following command can be used to create a conda environment named `instance-specific-param` with all the dependencies.
```bash
conda env create -f environment.yml
```

## Datasets
The dataset can be found at:

* [SIDD](https://abdokamel.github.io/sidd/)


We used 23 scene instances captured using an iPhone 7 under normal lighting conditions with ISO settings greater than 200. From these instances, we extracted 150 random crops of size 512 × 512 pixels, yielding a total of 6,900 images. These should be kept in "SIDD_Crop" folder. The Train, Val and Test splits are obtained by running :
```bash
python SIDD_crop_bm3d.py
```

## Training

The following two scripts are used for training and evaluation our propsed Algorithm 2 and 3 

```bash
python dynamic_optimization.py
```
```bash
python instance_specific_param.py
```
