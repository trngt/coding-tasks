
# Analyzing EM Images of Mitochondria using DINOv3 Embeddings

## Overview

Advancements in electron microscopy (EM) technology have driven major breakthroughs in biomedical research. However, working with large imaging datasets generated from EM requires labor-intensive and accurate labeling. Here, we demonstrate the capabilities of a pre-trained self-supervised model for characterizing mitochondria in two datasets.

We leverage two datasets from HHMI-Janelia's OpenOrganelle project with mitochondria segmented by COSEM. We use Meta's pre-trained DINOv3 model to capture semantic mitochondria embeddings.

Using these embeddings, we can: (1) demonstrate DINOv3's capabilities of capturing semantic embeddings of mitochondria and (2) characterize embedding variation within a dataset and measure how that variation generalizes across datasets. Detailed write-up of methodology, results, and discussion is described in [`analysis.md`](https://github.com/trngt/coding-tasks/blob/main/task2/analysis.md).

## Setup

Basic setup uses a conda environment defined in `environment.yml`
```bash
conda create -f environment.yml
conda activate mito-mia
```

Ensure that the hugging face authentication token is loaded. This command will prompt the user for an authentication token with read access to the DINOv3 models on HuggingFace.
```
hf auth login
```

## Usage

Basic usage runs the entire python in the `mito_mia.py` file. This analysis runs against the mus-liver and mus-kidney datasets.

```bash
python mito_mia.py
```

## Project workflow

The general workflow (1) loads the data, (2) processes it into relevant mitochondrial embeddings, and (3) analyzes the embeddings across and within the datasets. Details are as follows:

1. Load DINOv3 vision model via HuggingFace.
2. Load OpenOrganelle datasets.
	For each dataset, compute slices of the volumetric data. Because of the 3D nature of the data, we partition the volume into 2D images across the z-axis.

	Then subset to relevant slices with whole mitochondria. Mitochondria are assigned to relevant slices for embedding computation. Remove slices from the set if they do not contain a mitochondrion.
3. Compute per-mitochondria embeddings.

	For each slice, compute patch-level embeddings. For each mitochondrion, retrieve the patch-level embeddings for its relevant slice and upsample to dense embeddings using bilinear interpolation.

	Using the COSEM segmentation labels, summarize each mitochondrion's embeddings into a single embedding vector as an average.

4. Analyse mitochondrial embeddings within and across datasets.

	With the embeddings, we can analyze each dataset by taking a single reference mitochondrion and evaluating the nearest and furthest mitochondria from the reference.

	Additionally, we can use this reference mitochondrion to evaluate how well the reference embedding generalizes to mitochondria in the second dataset.

##  File and Class structure

The project is organized into several components. The most important are described below:

- `mito_mia.py` - Entry point. Loads the model, runs both dataset pipelines, and runs the cross-dataset analysis. All figures are saved to `output/`.
- `DatasetPipeline / dataset_pipeline.py` - Orchestrates the end-to-end single-dataset workflow: data loading, slice generation, catalog building, embedding computation, and per-mito vector extraction. Call `run()` to execute all steps in sequence, or call individual steps (`load_data`, `generate_slices`, `build_catalog`, `compute_embeddings`, `build_mito_vectors`) independently.
- `ReferenceAnalyzer / reference_analyzer.py` - Within-dataset analysis. Given a fully run `DatasetPipeline`, selects a reference mitochondrion and computes L2 distances from its embedding vector to all other mitos in the same dataset. Produces distance distribution plots and thumbnail grids of the closest and furthest mitochondria.
- `CrossDatasetAnalyzer / cross_dataset_analyzer.py` - Cross-dataset analysis. Takes a source and target `DatasetPipeline`, uses the source's reference mito embedding, and ranks all mitos in both datasets by distance to that reference. Produces overlaid distance histograms and combined thumbnail grids color-coded by dataset of origin.
- `DataManager / data_manager.py` - Loads the EM and segmentation data from S3, keeping track of important attributes like  image resolution.
- `SliceGenerator / slice_generator.py` - Generates the initial even 2D slices of the EM data volume these images are of equal pre-defined sizes (e.g. 128x128, 224x224, 512x512).
- `MitoSliceManager / mito_slice_manager.py` - Creates a catalog of mitochondria segmentation IDs to relevant 2D slices. This manager keeps track of the final set of mitochondria to analyze within a data set.
- `EmbeddingsManager` / `embeddings.py` - Computes the embedding for a dataset at patch or dense resolution.
- `Visualizer` / `visualizer.py` - Contains various visualization code of the EM and segmentation data.
