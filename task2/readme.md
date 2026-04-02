
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

1. Load model

	- HF authentication

2. Load datasets
	- Compute slices
	- Subset to relevant slices (ones with mitochondria)
	- Create a mapping of mitochondria to slice:
		- Preserve the slice with the greatest coverage
		- Filter out small mitochondria (noisey)
			- Plot the distribution of mitochondria sizes

3. Per-mitochondria embeddings

	i. Compute embeddings for a single slice (decision point)
		- Allows for validation at the image-level, what the embedding distances look like for a mitochondria?
	ii. Compute embeddings for mitochondria (decision point)
		- Will still need to compute against 2D images.

4. Analysis:
	
	i. Validation: Mitochondria vs non-mitochondria:
		- Plot of selected mitochondria vs background of image
		- Plot of selected mitochondria vs other image

	i. Datasets as a whole:
		- (not Plotting all mitochondria Dimensionality reduction (PCA, UMAP, t-SNE)?

	ii. Compare single mitochondria to dataset
		- Distance/variation from this mitochondria?

	iii. Compare single mitochondria across data sets
		- Distance/variation from this mitochondria?


### Class structure

- Data manager
	- Loads the EM and segmentation data from S3.
- Slice generator
	- Generates the initial slicing of the data set volume into equal sized image patches (e.g. 512x512)
- Embeddings manager
	- Handles dataset-scale embedding compute at both full (dense) and patch resolution.
- New class ideas: 
	- Mitochondrial embeddings manager (computes per-mitochondria embeddings for a dataset)
	- Data Comparison Analyzer

- Visualizer (general visualization, can be split or focused into specific cases)

## Discussion

### Project Limitations

### Decisions

### Dense Embeddings

### Slicing and subsetting

### Image resolution and determining patch size

### Multiple Queries

### Proposal for Fine-tuning
