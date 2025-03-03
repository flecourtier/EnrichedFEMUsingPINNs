# EnrichedFEMUsingPINNs

<!-- This repo is in progress. -->

Arxiv link : ["Enriching continuous Lagrange finite element approximation spaces using neural networks"](https://arxiv.org/abs/2502.04947)

This repo contains all the material needed to run the various numerical results obtained.

## Documentation

A python documentation is available [here](https://flecourtier.github.io/EnrichedFEMUsingPINNs/build_sphinx/index.html).

## Installation

The python code is based on the following 2 main modules: 
- [ScimBa](https://gitlab.inria.fr/sciml/scimba) (based on pytorch): for creation of the prior (PINNs prediction)
- [FEniCS](https://fenicsproject.org/download/archive/): for finite element resolutions (and eventually mshr for mesh generation)  

We provide two ways of installing the necessary modules: the first via docker (https://docs.docker.com/engine/install/ubuntu/) and the second via anaconda (https://www.anaconda.com/docs/getting-started/anaconda/install).

### Docker

A docker image is available via the command :

```sh
docker pull flecourtier/enrichedfem:1.0.0
```

### Conda

* **Create the conda environment and install FEniCS:**
```sh
conda create -n enrichedfem -c conda-forge fenics mshr python=3.9.16
conda activate enrichedfem
conda update -n base -c defaults conda
```

* **Install Pytorch:**
```sh
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

* **Install requirements and enrichedfem module (in editable mode):**
To be executed at the root of the git repo.
```sh
pip install -r requirements.txt
pip install -e .
```

* **Install ScimBa (in editable mode):**
```sh
git clone https://gitlab.inria.fr/sciml/scimba.git
cd scimba
pip install -e .
```