
# Analyzing EM Images of Mitochondria using DINOv3 Embeddings

## Overview

Advancement in electron microscopy (EM) technology have driven major breakthroughs in biomedical research. However, working large imaging datasets generated from EM require labor-intensive and accurate labeling. Here, we demonstrate the capabilities of a pre-trained self-supervised model for characterizing mitochondria in two datasets.

We leverage two datasets from HHMI-Janelia's OpenOrganelle project with mitochondria segmented by COSEM. We use Meta's pre-trained DINOv3 model's to capture semantic mitochondria embeddings. 

Using these embeddings, we can: (1) demonstrate DINOv3's capabilities of capturing semantic embeddings of mitochondria and (2) analyze within data set variation and compare that variation to a second dataset. <mark>**todo: find a more clear way to develop the second idea.**</mark>

## Project workflow

The general workflow (1) loads the data, (2) processes it into relevant mitochondrial embeddings, and (3) analyzes the embeddings across and within the datasets. Details are as follows:

1. Load DINOv3 vision model <mark>todo: via hugging face API, currently using a local copy</mark>
2. Load OpenOrganelle datasets.
	For each dataset, compute slices of the volumetric data. Because of the 3D nature of the data, we slice the data as partitioned 2D images across the z-plane.

	then subset to relevant slices with whole mitochondria. Mitochondrion are assigned to relevant slices, for eventual embeddings computation. Remove slices from the set if they do not contain a mitochondrion.
3. Compute per-mitochondria embeddings. 
	
	For each slice, compute per patch-level embeddings. For each mitochondrion, retrieve the patch-level embeddings for its relevant slice and upsample to dense embeddings using bilinear interpolation <span></mark>todo: here, we'll need to incorporate/add in the dense embeddings justification/step.</mark>

	Using the COSEM segmentation labels, summarize each mitochondrion's embeddings into a single embeddings vectors as an average.

4. Analyse mitochondrial embeddings within and across datasets
	
	With the embeddings, we can analyze each dataset taking a single reference mitochondrion, and evaluating the nearest and furthest mitochondrion from the reference.

	Additionally, we can use this reference mitochondrion to assess the generalized embeddings onto the second dataset's mitochondria embeddings.

##  File and Class structure

The project is organized into several components. The most important are described below:

- DataManager `data_manager.py` - Loads the EM and segmentation data from S3, keeping track of important attributes like  image resolution.
- Slice generator `slice_generator.py` - Generates the initial even 2D slices of the EM data volume these images are of equal pre-defined sizes (e.g. 128x128, 224x224, 512x512).
- MitoSliceManager `mito_slice_manager.py` - Creates a catalog of mitochondria segmentation IDs to relevant 2D slices. This manager keeps track of the final set of mitochondria to analyze within a data set.
- Embeddings manager `embeddings.py` - Computes the embedding for a dataset at patch or dense resolution.


- Visualizer (general visualization, can be split or focused into specific cases)

## Discussion

### Project Limitations

### Decisions

### Dense Embeddings

### Slicing and subsetting

### Image resolution and determining patch size

### Multiple Queries

### Proposal for Fine-tuning
