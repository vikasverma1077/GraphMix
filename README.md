# GraphMix
This is the Pytorch code for reproducing the results of the Paper GraphMix (https://arxiv.org/pdf/1909.11715.pdf). GraphMix is a simple and efficient regularization method for Graph Neural Networks (GNN) based Semi-Supervised Learning over Graph Structured Data. We show that with this regularizer, even the simpler GNN architecture such as GCN (Kipf et. al.) can achiever state-of-the-art results on benchmark graph datasets such as Cora/Citeseer/Pubmed.

## Requirements 
This code is tested with Python 3.6.3 and requires following packages (see requirements.txt list of all the packages):

torch==1.1.0

numpy==1.16.3

pandas==0.24.1

Pillow==5.3.0

scikit-learn==0.21.2

scipy==1.2.1

seaborn==0.9.0

six==1.12.0

tqdm==4.32.2


# How to run 

For reproducing results of GraphMix(GCN) of Table1 in the paper, go to directory GraphMix/semisupervised/codes and run the following commands:

`python run_cora.py`

`python run_citeseer.py`

`python run_pubmed.py`


This codebase is based on the github repo : https://github.com/DeepGraphLearning/GMNN

To all the people using this codebase: let us know in the "issues" if you are having some difficulity in reproducing the results. 


