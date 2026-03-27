
## Current Notes and Ideas

### Determine appropriate patch size with justification:
Both the ViT (Dosovitskiy, 2020) and DINOv2 (Oquab, 2024) papers describe the compute and efficiency trade-off with smaller patch sizes. Oquab describes this through image resolution comparisons. 

For a comprehensive analysis, I will look at the microscopy data and examine:
1. The image resolution of the dataset and
2. The size of the mitochondria

This exploration will motivate the size of the patches. I am interested in sizing the patches to capture enough of the mitochondrial size patches per mitochondria. Will 2x2 or 3x3 patches per mitochondria be small enough?

Current status: Explore a microscopy data set.

### Dense embeddings
The other question I am working through is: how are dense embeddings generated from the patched segmentation output of DINO. In the Oquab DINOv2 paper, two methods were applied: (1) a bilinear interpolation, and (2) a boosted multiscale test-time augmentation. The latter is a bit more complex, but may be worth digging into. The authors describe the bilinear sampling as incapable of producing high-resolution segmentations.

Current status: Continue exploration of this implementation.

### Querying and Labeling
I have an ongoing question, prior to exploring the data. How are mitochondria labeled? Are they labeled beforehand in the dataset? Or do we have enough information following DINO segmentation. The former is the logical answer. Then, I will need to understand the process of processing this data and its annotations.

Current status: Explore a microscopy data set.

