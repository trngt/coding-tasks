
# Analyzing EM Images of Mitochondria using DINOv3 Embeddings

Advancement in electron microscopy (EM) technology have driven major breakthroughs in biomedical research. However, working large imaging datasets generated from EM require labor-intensive and accurate labeling. Here, we demonstrate the capabilities of a pre-trained self-supervised model for characterizing mitochondria in two datasets.

We leverage two datasets from HHMI-Janelia's OpenOrganelle project with mitochondria segmented by COSEM. We use Meta's pre-trained DINOv3 model's to capture semantic mitochondria embeddings. 

Using these embeddings, we can: (1) demonstrate DINOv3's capabilities of capturing semantic embeddings of mitochondria and (2) analyze within data set variation and compare that variation to a second dataset. **todo: find a more clear way to develop the second idea.**

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
