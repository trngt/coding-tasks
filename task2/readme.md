
## Current Notes and Ideas

### Determine appropriate patch size with justification:

Patch size is determined by the mitochondria size and the microscopy image resolution (nm per pixels). We want the image patch sizes to capture whole mitochondria.

Two definitions of patch sizes to make clear:
1. Image patch size: Image size for processing with DINOv3 (e.g. 224x224 or 128x128 px). This is what we are interested in.
2. DINOv3 patch size: The sub-patches the model will slice the input images into. Fixed at 16 px.

Unless specified, patch size will refer to (1).

```
Steps:
1. Determine average mitochondria size [(Alberts B, 2002)](https://www.ncbi.nlm.nih.gov/books/NBK26894/)
	0.5-1 microns
2. Determine pixel resolution (nm / pixel) from the attributes.json for the appropriate em n5 dataset:
	For s0, the resolution is 8x8x8 nm / pixel
3. Then, compute average mitochondria size per pixel.
	(0.5-1)*(1000)/(8) = 62.5-125 pixels
4. Thus at, s0 (8x8x8 resolution), the average mitochondria size will be
    62.5-125
At s1 (16x16x16), they will be:
	31.25-62.5
5. An appropriate data set patch size will capture the mitochondria at pixel scales:
	224x	224 px at s0
and 
	128x128 px at s1
```

This calculation will work at different scales. The decision for the patch size is determined by the DINOv3 patch size, fixed at 16x16. That is 224 and 128 pixel dimesion are both divisible by model's 16 px patch size.


#### Related to the image resolution: 
Both the ViT (Dosovitskiy, 2020) and DINOv2 (Oquab, 2024) papers describe the compute and efficiency trade-off with smaller patch sizes. Oquab describes this through image resolution comparisons. 

---

### Dense embeddings
The other question I am working through is: how are dense embeddings generated from the patched segmentation output of DINO. In the Oquab DINOv2 paper, two methods were applied: (1) a bilinear interpolation, and (2) a boosted multiscale test-time augmentation. The latter is a bit more complex, but may be worth digging into. The authors describe the bilinear sampling as incapable of producing high-resolution segmentations.

Current status: Continue exploration of this implementation.

### Querying and Labeling
I have an ongoing question, prior to exploring the data. How are mitochondria labeled? Are they labeled beforehand in the dataset? Or do we have enough information following DINO segmentation. The former is the logical answer. Then, I will need to understand the process of processing this data and its annotations.

Current status: Explore a microscopy data set.
