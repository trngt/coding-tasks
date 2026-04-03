
# Analyzing EM Images of Mitochondria using DINOv3 Embeddings

Advancement in electron microscopy (EM) technology have driven major breakthroughs in biomedical research. However, working large imaging datasets generated from EM require labor-intensive and accurate labeling. Here, we demonstrate the capabilities of a pre-trained self-supervised model for characterizing mitochondria in two datasets.

We leverage two datasets from HHMI-Janelia's OpenOrganelle project with mitochondria segmented by COSEM. We use Meta's pre-trained DINOv3 model's to capture semantic mitochondria embeddings. 

Using these embeddings, we can: (1) demonstrate DINOv3's capabilities of capturing semantic embeddings of mitochondria and (2) analyze within data set variation and compare that variation to a second dataset. <mark>**todo: find a more clear way to develop the second idea.**</mark>

## Project structure

### Workflow

1. Load DINOv3 vision model <mark>todo: via hugging face API, currently using a local copy</mark>

2. Load OpenOrganelle datasets.
	For each dataset, compute slices of the volumetric data. Because of the 3D nature of the data, we slice the data as partitioned 2D images across the z-plane.

	then subset to relevant slices with whole mitochondria. Mitochondrion are assigned to relevant slices, for eventual embeddings computation. Remove slices from the set if they do not contain a mitochondrion.

3. Compute per-mitochondria embeddings. 
	
	For each slice, compute per patch-level embeddings. For each mitochondrion, retrieve the patch-level embeddings for its relevant slice and upsample to dense embeddings using bilinear interpolation <span></mark>todo: here, we'll need to incorporate/add in the dense embeddings justification/step.</mark>

	Using the COSEM segmentation labels, summarize each mitochondrion's embeddings into a single embeddings vectors as an average.

4. Analysis:
	
	With the embeddings, we can analyze each dataset taking a single reference mitochondrion, and evaluating the nearest and furthest mitochondrion from the reference.

	Additionally, we can use this reference mitochondrion to assess the generalized embeddings onto the second dataset's mitochondria embeddings.

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
