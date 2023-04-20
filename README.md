# MultiETM: CPSC536R course project

MultiETM is an extension of scETM to multimodal data.
In this respository, we have included the MultiETM model and experiments on CITE-seq data.
The experiments on the naive concat-scETM model are also included.

The experiments are configured in the `experiments` directory.
They are YAML files with the following format:

```yaml
files:
  rna:
    - path to h5ad file for batch 1 RNA data
    - path to h5ad file for batch 2 RNA data
    - ...
  protein:
    - path to h5ad file for batch 1 protein data
    - path to h5ad file for batch 2 protein data
    - ...
  combined:
    - path to h5ad file for batch 1 concatenated RNA and protein data
    - path to h5ad file for batch 2 concatenated RNA and protein data
    - ...
  gene_indices: Python pickle file containing array of genes to zero out during imputation. Only needed for imputation experiments.
model_params:
  n_epochs: number of epochs to train
  eval_every: number of epochs at which the model should be evaluated
  cell_type_col: name of column in AnnData obs that contains the cell type annotation. None if there are no annotations
  day: for my use only, can be anything
  donor: for my use only, can be anything
  rna_n_vars: number of genes in the RNA data
ckpt_dir: directory to store all training information
```

All the notebooks needed to run the different models and experiments are located in the `notebooks` directory.
They are:

* `base-scETM.ipynb`: trains and evaluates the scETM model on only RNA data
* `concat-scETM.ipynb`: trains and evaluates concat-scETM (which concatenates the RNA and protein data and uses scETM as-is)
* `MultiETM.ipynb`: trains and evaluates the MultiETM model
* `Gene Imputation.ipynb`: experiment to train a model and impute zeroed out genes
* `Scanpy Analysis.ipynb` experiment to evaluate ARI and NMI clustering metrics using PCA on only RNA and on concatenated RNA and protein data.
* `Muon WNN Analysis.ipynb`: experiment to evaluate ARI and NMI clustering metrics using weighted nearest neighbours as implemented in Muon.
* `Pathway Analysis.ipynb`: using the saved linear decoder matrices, find biological pathways associated with each topic.

The h5ad files used for the experiments are available upon request because I do not know the legality/ethics of uploading the derivative h5ad data from other experiments (anonymized data, attribution, etc.).
The datasets used for experiments were:

* https://www.kaggle.com/competitions/open-problems-multimodal/data
* https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-10026
* https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-10129 (this one went unused in the project)


# scETM: single-cell Embedded Topic Model
A generative topic model that facilitates integrative analysis of large-scale single-cell RNA sequencing data.

The full description of scETM and its application on published single cell RNA-seq datasets are available [here](https://www.biorxiv.org/content/10.1101/2021.01.13.426593v1).

This repository includes detailed instructions for installation and requirements, demos, and scripts used for the benchmarking of 7 other state-of-art methods.


## Contents ##

- [scETM: single-cell Embedded Topic Model](#scetm-single-cell-embedded-topic-model)
  - [Contents](#contents)
  - [1 Model Overview](#1-model-overview)
  - [2 Installation](#2-installation)
  - [3 Usage](#3-usage)
    - [Data format](#data-format)
    - [Number of training epochs](#number-of-training-epochs)
    - [A taste of scETM](#a-taste-of-scetm)
    - [p-scETM](#p-scetm)
    - [Transfer learning](#transfer-learning)
      - [Pre-aligned datasets](#pre-aligned-datasets)
      - [Unaligned datasets](#unaligned-datasets)
    - [Tensorboard Integration](#tensorboard-integration)
  - [4 Benchmarking](#4-benchmarking)

## 1 Model Overview

![](doc/scETM.png "scETM model overview")
**(a)** Probabilistic graphical model of scETM. We model the scRNA-profile read count matrix y<sub>d,g</sub> in cell d and gene g across S subjects or studies by a multinomial distribution with the rate parameterized by cell topic mixture θ, topic embedding α, gene embedding ρ, and batch effects λ. **(b)** Matrix factorization view of scETM. **(c)** Encoder architecture for inferring the cell topic mixture θ.

## 2 Installation
Python version: 3.7+
scETM is included in PyPI, so you can install it by

```bash
pip install scETM
```

To enable GPU computing (which significantly boosts the performance), please install [PyTorch](https://pytorch.org/) with GPU support **before** installing scETM.

## 3 Usage
**A step-by-step scETM tutorial can be found in [here](/notebooks/scETM%20introductory%20tutorial.ipynb).**

### Data format
scETM requires a cells-by-genes matrix `adata` as input, in the format of an AnnData object. Detailed description about AnnData can be found [here](https://anndata.readthedocs.io/en/latest/).

By default, scETM looks for batch information in the 'batch_indices' column of the `adata.obs` DataFrame, and cell type identity in the 'cell_types' column. If your data stores the batch and cell type information in different columns, pass them to the `batch_col` and `cell_type_col` arguments, respectively, when calling scETM functions.

### Number of training epochs
Note that scETM is trained via batched gradient descent. For small datasets (e.g. MousePancreas which contains 1886 cells) and large mini-batch sizes (e.g. 2000), training one epoch generates only one gradient step (i.e. updates model parameter only once) as the whole dataset can be fit into a single mini-batch. In these cases, it is recommended to set the number of training epochs such that there is at least 6k gradient updates to the model. Otherwise, the model may not have converged when training stops.

### A taste of scETM

```python
from scETM import scETM, UnsupervisedTrainer, evaluate
import anndata

# Prepare the source dataset, Mouse Pancreas
mp = anndata.read_h5ad("MousePancreas.h5ad")
# Initialize model
model = scETM(mp.n_vars, mp.obs.batch_indices.nunique(), enable_batch_bias=True)
# The trainer object will set up the random seed, optimizer, training and evaluation loop, checkpointing and logging.
trainer = UnsupervisedTrainer(model, mp, train_instance_name="MP", ckpt_dir="../results")
# Train the model on adata for 12000 epochs, and evaluate every 1000 epochs. Use 4 threads to sample minibatches.
trainer.train(n_epochs=12000, eval_every=1000, n_samplers=4)
# Obtain scETM cell, gene and topic embeddings. Unnormalized cell embeddings will be stored at mp.obsm['delta'], normalized cell embeddings at mp.obsm['theta'], gene embeddings at mp.varm['rho'], topic embeddings at mp.uns['alpha'].
model.get_all_embeddings_and_nll(mp)
# Evaluate the model and save the embedding plot
evaluate(mp, embedding_key="delta", plot_fname="scETM_MP", plot_dir="figures/scETM_MP")
```

### p-scETM
p-scETM is a variant of scETM where part or all of the the gene embedding matrix ρ is fixed to a pathways-by-genes matrix, which can be downloaded from the [pathDIP4 pathway database](http://ophid.utoronto.ca/pathDIP/Download.jsp). We only keep pathways that contain more than 5 genes.

If it is desired to fix the gene embedding matrix ρ during training, let trainable_gene_emb_dim be zero. In this case, the gene set used to train the model would be the intersection of the genes in the scRNA-seq data and the genes in the gene-by-pathway matrix. Otherwise, if trainable_gene_emb_dim is set to a positive value, all the genes in the scRNA-seq data would be kept.

### Transfer learning

#### Pre-aligned datasets

In this setting, the source and target dataset are pre-aligned, meaning they have the same / homologous gene **lists** (note that the order of the genes would have to be aligned).

```python
from scETM import scETM, UnsupervisedTrainer, prepare_for_transfer
import anndata

# Prepare the aligned source dataset, Mouse Pancreas
mp = anndata.read_h5ad("MousePancreas_aligned.h5ad")
# Load the aligned target dataset, Human Pancreas
hp = anndata.read_h5ad('HumanPancreas_aligned.h5ad')
# Initialize model
model = scETM(mp.n_vars, mp.obs.batch_indices.nunique(), enable_batch_bias=True)
# The trainer object will set up the random seed, optimizer, training and evaluation loop, checkpointing and logging.
trainer = UnsupervisedTrainer(model, mp, train_instance_name="MP", ckpt_dir="../results")
# Train the model on adata for 12000 epochs, and evaluate every 1000 epochs. Use 4 threads to sample minibatches.
trainer.train(n_epochs=12000, eval_every=1000, n_samplers=4)
# Directly apply MP-trained model on HP, storing cell embeddings to hp.obsm['delta'] (zero-shot transfer).
model.get_cell_embeddings_and_nll(hp, emb_names='delta')
# Evaluate the model and save the embedding plot
evaluate(hp, embedding_key="delta", plot_fname="scETM_MP_to_HP", plot_dir="figures/scETM_transfer")
# Optionally, instantiate another trainer to fine-tune the model
trainer = UnsupervisedTrainer(model, hp, train_instance_name="HP_MPtransfer", ckpt_dir="../results", init_lr=5e-4)
trainer.train(n_epochs=800, eval_every=200)
```

#### Unaligned datasets

In this setting, a model is trained for the source dataset, and then adapted to (fine-tuned on) the target dataset, which could have different gene sets. The `prepare_for_transfer` function discards the parameters tied to the source-dataset-unique genes and randomly initializes those tied to the target-dataset-unique genes. Because of this significant change, the model would usually require fine-tuning before used to evaluate the target dataset.

```python
from scETM import scETM, UnsupervisedTrainer, prepare_for_transfer
import anndata

# Prepare the source dataset, Mouse Pancreas
mp = anndata.read_h5ad("MousePancreas.h5ad")
# Initialize model
model = scETM(mp.n_vars, mp.obs.batch_indices.nunique(), enable_batch_bias=True)
# The trainer object will set up the random seed, optimizer, training and evaluation loop, checkpointing and logging.
trainer = UnsupervisedTrainer(model, mp, train_instance_name="MP", ckpt_dir="../results")
# Train the model on adata for 12000 epochs, and evaluate every 1000 epochs. Use 4 threads to sample minibatches.
trainer.train(n_epochs=12000, eval_every=1000, n_samplers=4)

# Load the target dataset, Human Pancreas
hp = anndata.read_h5ad('HumanPancreas.h5ad')
# Align the source dataset's gene names (which are mouse genes) to the target dataset (which are human genes)
mp_genes = mp.var_names.str.upper()
mp_genes.drop_duplicates(inplace=True)
# Generate a new model and a modified dataset from the previously trained model and the mp_genes
model, hp = prepare_for_transfer(model, hp, mp_genes,
	  keep_tgt_unique_genes=True,    # Keep target-unique genes in the model and the target dataset
    fix_shared_genes=True          # Fix parameters related to shared genes in the model
)
# Instantiate another trainer to fine-tune the model
trainer = UnsupervisedTrainer(model, hp, train_instance_name="HP_all_fix", ckpt_dir="../results", init_lr=5e-4)
trainer.train(n_epochs=800, eval_every=200)
```

### Tensorboard Integration

If a Tensorboard SummaryWriter is passed to the `writer` argument of the `UnsupervisedTrainer.train` method, the trainer will automatically log cell, gene and topic embeddings to a tensorboard logdir. The gene and topic embeddings are in the same space.

![tb_cell](doc/tensorboard_cell.png)
![tb_gene_topic](doc/tensorboard_genetopic.png)

## 4 Benchmarking
The commands used for running [Harmony](https://github.com/immunogenomics/harmony), [Scanorama](https://github.com/brianhie/scanorama), [Seurat](https://satijalab.org/seurat/), [scVAE-GM](https://github.com/scvae/scvae), [scVI](https://github.com/YosefLab/scvi-tools), [LIGER](https://github.com/welch-lab/liger), [scVI-LD](https://www.biorxiv.org/content/10.1101/737601v1.full.pdf) are available in the [scripts](/scripts) folder.
