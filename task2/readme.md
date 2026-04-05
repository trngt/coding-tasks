
# Analyzing EM Images of Mitochondria using DINOv3 Embeddings

## Overview

Advancements in electron microscopy (EM) technology have driven major breakthroughs in biomedical research. However, working with large imaging datasets generated from EM requires labor-intensive and accurate labeling. Here, we demonstrate the capabilities of a pre-trained self-supervised model for characterizing mitochondria in two datasets.

We leverage two datasets from HHMI-Janelia's OpenOrganelle project with mitochondria segmented by COSEM. We use Meta's pre-trained DINOv3 model to capture semantic mitochondria embeddings.

Using these embeddings, we can: (1) demonstrate DINOv3's capabilities of capturing semantic embeddings of mitochondria and (2) characterize embedding variation within a dataset and measure how that variation generalizes across datasets.

## Project workflow

The general workflow (1) loads the data, (2) processes it into relevant mitochondrial embeddings, and (3) analyzes the embeddings across and within the datasets. Details are as follows:

1. Load DINOv3 vision model <mark>todo: via hugging face API, currently using a local copy</mark>
2. Load OpenOrganelle datasets.
	For each dataset, compute slices of the volumetric data. Because of the 3D nature of the data, we partition the volume into 2D images across the z-axis.

	Then subset to relevant slices with whole mitochondria. Mitochondria are assigned to relevant slices for embedding computation. Remove slices from the set if they do not contain a mitochondrion.
3. Compute per-mitochondria embeddings.

	For each slice, compute patch-level embeddings. For each mitochondrion, retrieve the patch-level embeddings for its relevant slice and upsample to dense embeddings using bilinear interpolation. <mark>todo: incorporate the dense embeddings justification/step.</mark>

	Using the COSEM segmentation labels, summarize each mitochondrion's embeddings into a single embedding vector as an average.

4. Analyse mitochondrial embeddings within and across datasets.

	With the embeddings, we can analyze each dataset by taking a single reference mitochondrion and evaluating the nearest and furthest mitochondria from the reference.

	Additionally, we can use this reference mitochondrion to evaluate how well the reference embedding generalizes to mitochondria in the second dataset.

##  File and Class structure

The project is organized into several components. The most important are described below:

- `DataManager / data_manager.py` - Loads the EM and segmentation data from S3, keeping track of important attributes like  image resolution.
- `SliceGenerator / slice_generator.py` - Generates the initial even 2D slices of the EM data volume these images are of equal pre-defined sizes (e.g. 128x128, 224x224, 512x512).
- `MitoSliceManager / mito_slice_manager.py` - Creates a catalog of mitochondria segmentation IDs to relevant 2D slices. This manager keeps track of the final set of mitochondria to analyze within a data set.
- `EmbeddingsManager` / `embeddings.py` - Computes the embedding for a dataset at patch or dense resolution.
- `Visualizer` / `visualizer.py` - Contains various visualization code of the EM and segmentation data.

## Analysis

### Validation of mitochondrial embeddings

### Within-data set single mitochondrion analysis

### Across data set mitochondrian analysis

## Discussion

### Slicing and subsetting

- Computing resources limitation (M1 Pro Laptop with 16 GB of Memory)
	- Limits analysis to representative sample of slices and mitochondria per dataset.
- Mitochondrial embeddings may be distorted at boundaries of image slices.

### Producing dense embeddings

There are two relevant approaches that I've found to produce dense embeddings, with training and without training. In the DINOv2 paper (Oquab, 2021), the authors train a linear and SOTA pipeline for semantic segmentation. However, these approaches require the setup of a training pipeline specifically for computing these embeddings. 

For this, proof-of-concept work, I've chosen to follow the approach described in (González-Marfil, 2025).
In their work, they demonstrate accurate semantic segmentation masks for microscopy imaging from this simple interpolation approach.

I apply a bilinear interpolation directly from the patch embeddings to scale the final per-slice embeddings to the input resolution. Given, the resolution of my patches relative the mitochondria size, this has approach has been adequate for this first iteration of the project.

### Determining patch size

- Fixed patch size defined by DINOv3's Vision models.
- Image resolution and image size is configurable.
- Trade-off: high-resolution embeddings and computational cost
- Capture mitochondria by size (0.5-1 microns). 
	- Multiple patches ~8-16 patches suitable for capturing ultrastructure.

### Multiple Queries

- Use case: capture variation within datasets
- Meta-mitochondria that captures a robust representative per-dataset mitochondria. <mark>Continue building out ideas here.</mark>

### Proposal for Fine-tuning

- LoRA: Freeze DINOv3 weights and add low-rank adapters. Strongly supported in literature: large complex models are prime for quick and accurate fine-tuning with very small adapaters and few trainable weights for specific downstream tasks. <mark>Evaluate literature on LoRA details.</mark>
- Phase 1: DINO, SSL training on large labeled and unlabeled dataset.
- Phase 2: Continue training on labeled dataset to accurately segment known mitochondria. <mark>Evaluate literature on fine-tuning training with labeled and unlabeled data</mark>
- Evaluation: Evaluate quality of fine-tuning using known labels. Assess visually against new data sets <mark>Revist literature for SSL model evaluation.</mark>

