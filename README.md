NIPS'14-SSL
==========

Code for reproducing some key results of our NIPS 2014 paper on semi-supervised learning (SSL) with deep generative models.

D.P. Kingma, D.J. Rezende, S. Mohamed, M. Welling  
**Semi-Supervised Learning with Deep Generative Models**  
Advances in Neural Information Processing Systems 27 (**NIPS 2014**), Montreal  
[http://arxiv.org/abs/1406.5298](http://arxiv.org/abs/1406.5298)

Please cite this paper when using this code for your research.

_Warning_: This code is far from fully commented.

For questions and bug reports, please send me an e-mail at _dpkingma[at]gmail.com_.

## Prerequisites

1. Make sure that recent versions installed of:
	- Python (version 2.7 or higher)
	- Numpy (e.g. `pip install numpy`)
	- Theano (e.g. `pip install Theano`)

2. Set `floatX = float32` in the `[global]` section of Theano config (usually `~/.theanorc`). Alternatively you could prepend `THEANO_FLAGS=floatX=float32 ` to the python commands below. 

3. Clone this repository, e.g.:
```sh
git clone https://github.com/dpkingma/nips14-ssl.git
```

4. Set an environment variable `ML_DATA_PATH` that points to subdirectory `data/`. For example, if you checked out this repo to your home directory:   
```sh
export ML_DATA_PATH="$HOME/nips14-ssl/data"
```

# Qualitative results

## Flying through latent space of M2 model
To generate movies of flying through latent-space of the M2 model, run:
```sh
python run_flying.py [dataset] 1 output.mkv
```
where `dataset` is 'mnist' or 'svhn', and `target_filename` is the filename to save the movie file to. NOTE: This script requires ffmpeg to be installed.

## Analogies
Run:
```sh
python run_analogies.py [dataset] 1
```

# Quantitative results

## Learning M1 model

To train model M1 (a standard Variational Auto-Encoder / DLGM with sperical Gaussian latent space):
```sh
python run_gpulearn_z_x.py [dataset]
```
The M1 model does not incorporate class label, but is used in the paper's experiments for feature extration.

## Learning M1+M2 model, partially observed labels

To run the semi-supervised learning experiments with model M1+M2:
```sh
python run_2layer_ssl.py [n_labels] [seed]
```
where `n_labels` is the number of labels, and `seed` is the random seed for Numpy. To reproduce the experimental results in the paper, the number of labels should be in (100,600,1000,3000). The random seed can be any integer. Each experiment will run for 3000 epochs; since this code is not GPU-optimized, running many epochs might take a few days to complete. However, it is often not necessary to run the the algorithm for so many epochs to produce good results.

## Learning M2 model with fully observed labels

For training a generative model with all labels:
```sh
python run_gpulearn_yz_x.py [dataset]
```
where `dataset` is 'mnist', 'svhn', 'norb' or 'norb_reshuffled'.


## Evaluate test-set error of models trained with all labels

For evaluating the test-set classification error using already trained generative models of MNIST and SVHN:
```sh
python run_sl.py [dataset]
``` 
This iteratively builds, for each test-set image, an importance-sampled estimate of the posterior probability distribution over the class labels. This is an expensive procedure, but may be speed up by using fitting an inference model to the posterior distribution of class labels (which wasn't done in this case).




