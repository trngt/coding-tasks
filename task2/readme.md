
# Analyzing EM Images of Mitochondria using DINOv3 Embeddings

## Overview

### 1. What is EM and what it help us learn?

- Advancements: Mapping connectome
- Imaging structures of proteins revealing cellular dynamics in 3D space

### 2. Brief overview of the problem.

- EM datasets are large and labor-intensive to label.
- Advancements have been made segmented large volumentric datasets of organelles (COSEM).
	- Uses 3D convolutional neural networks (U-Nets).
- However, these labels require large amounts of labeled data.
- And may not capture semantically meaningful structures of organelle, creating challenges in generalizing to unseen data.

### 2. Data sets and Models

- We can use the existing datasets (COSEM) where mitochondria were segmented to drive analysis segmentation modeling using SSL methods, DINOv3.
- DINOv3, SSL ViT model that captures rich general semantic embeddings, trained on X datasets.


### 2. What we can do at a high-level.

- We can apply DINOv3, to capture the semantic embeddings of labeled mitochondrial data.
- Then analyze how well these embeddings capture the semantic ultrastructure of mitochondria within a dataset and across multiple datasets.


## Project structure

### Workflow

1. Load datasets
2. Load model
3. Compute embeddings
4. Compute mitochondrial embeddings
5. Compare mitochondria
6. Compare mitochondria across data sets

### Class structure

## Discussion

## Project Limitations

## Image resolution and determining patch size

## Multiple Queries

## Proposal for Fine-tuning
