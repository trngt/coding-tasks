
# Analysis

## Methodology

### Determining patch size and image resolution

Patch sizes were fixed to 16x16 as defined by the available DINOv3 ViT models. However, image resolution was configurable. The microscopy image data was available and several resolutions.

I selected the resolution based on available computing power and adequate patch coverage per mitochondria. By choosing a higher resolution EM dataset, more patches would cover each mitochondria, enabling greater identification of mitochondrial ultrastructure.

Based on these constraints I selected image resolutions in which 8-16 patches covered each mitochondria. With each mitochondria estimated as 0.5-1 microns (Alberts B, 2002), I chose image resolutions of around 16 nm per voxel, equating to around 31-62 pixel mitochondria along each axis.

### Data set slicing and subsetting

With limited computational resources, I chose to subsample the full volumetric data to a small random sample of each microscopy dataset.

The volumetric data was sliced into equal 512x512 px images separated along 128 px strides along the z-axis. Random slices of the volumetric data were selected. Mitochondria were assigned to the single slice, for which it is maximally represented (in pixels).

### Producing dense embeddings

There are two high-level relevant approaches that I've found to produce dense embeddings, with training and without training. In the DINOv2 paper (Oquab, 2023), the authors train a linear and SOTA pipeline for semantic segmentation. However, these approaches require the setup of a training pipeline specifically for computing these embeddings. 

For this, proof-of-concept work, I've chosen to follow the approach described in (González-Marfil, 2025).
In their work, they demonstrate accurate semantic segmentation masks for microscopy imaging from this simple interpolation approach.

I apply a bilinear interpolation directly from the patch embeddings to scale the final per-slice embeddings to the input resolution. Given, the resolution of my patches relative the mitochondria size, this has approach has been adequate for this first iteration of the project.

### Computing mitochondrial embeddings

With the dense, pixel-level embeddings computed, I calculated the average embeddings vector for each mitochondria.

The segmentation masks allowed me to select the relevant embeddings (on a per-pixel level), then take an average to compute a single vector embedding for each mitochondrion.

## Analysis

Following calculation of per-mitochondrion embeddings. I selected a single mitochondrion as a reference vector. This reference mitochondrion is then used to evaluate within data-set homegeneity and compare across against mitochondria within another dataset.

### Validation of mitochondrial embeddings

To validate the selected mitochondrial embeddings, I applied a simple metric to analyze the pixel distances within a microscopy slice. For each pixel embedding in the slice, I plot the distance to the reference mitochondria as a heatmap.

<div align="center">
	<img src="https://github.com/trngt/coding-tasks/blob/main/task2/figures/1_mus-liver_selected.png" width="70%"/>
  <p>Figure 1. Raw EM data (left), the selection of the reference mitochondrion (middle), and the per-pixel distances to the reference mitochondrion (right).</p>
</div>

Reassuringly, the boundaries of the mitochondria in within the slice were identified. However, there appears to a be locality bias to the distance calculations. Areas closer to the reference mitochondrion also appear closer in distance the reference mitochondrion embeddings. In future work, we can design a methodology to normalize for this locality bias. <mark>todo: develop ideas for normalization</mark>

### Within-data set single mitochondrion analysis

After selecting a reference mitochondrion, the distance to every other mitochondria in the dataset was calculated. This allowed me to evaluate the nearest and furthest mitochondrion from the reference. The embeddings appears to characterize the high-level shape and size of the selected mitochondria.

<div align="center">
	<img src="https://github.com/trngt/coding-tasks/blob/main/task2/figures/2_mus-liver_distances.png" width="70%"/>
  <p>Figure 2. Dataset distribution of distances to reference mitochondrion</p>
</div>

The 10 nearest mitochondrion appeared to show similar shape and structure to the reference.

<div align="center">
	<img src="https://github.com/trngt/coding-tasks/blob/main/task2/figures/3_mus-liver_closest_10.png" width="70%"/>
  <p>Figure 3. 10 nearest mitochondria.</p>
</div>

The 10 furthest, however, show much more diverse array of shapes and sizes. No relevant pattern within the structures were discerned across the closest and furthest mitochondria.
 
<div align="center">
	<img src="https://github.com/trngt/coding-tasks/blob/main/task2/figures/4_mus-liver_furthest_10.png" width="70%"/>
  <p>Figure 4. 10 furthest mitochondria.</p>
</div>

### Across data set mitochondrian analysis

When comparing across two data sets (mus-liver and mus-kidney), we see that the kidney mitochondrion are distributed closer on average to the selected mitochondrion in the liver dataset.

If the segmentation embeddings were biased towards the selected mitochondrion dataset, we would expect the closest mitochondria to be solely in the liver data set. 

However, the 20 closest and 20 furthest mitochondrion across both datasets shows a diversity of distances (random distribution of red:liver and blue:kidney mitochondrion).

<div align="center">
	<img src="https://github.com/trngt/coding-tasks/blob/main/task2/figures/5_combined_distances.png" width="70%"/>
  <p>Figure 5. 10 Distances to reference mitochondrion across both datasets.</p>
</div>


<div align="center">
	<img src="https://github.com/trngt/coding-tasks/blob/main/task2/figures/6_combined_closest_20.png" width="70%"/>
  <p>Figure 6. 20 closest mitochondria across both datasets.</p>
</div>

<div align="center">
	<img src="https://github.com/trngt/coding-tasks/blob/main/task2/figures/7_combined_furthest_20.png" width="70%"/>
  <p>Figure 7. 20 furthest mitochondria across both datasets.</p>
</div>

## Discussion

### Slicing and subsetting

Because of computing resource constraints (M1 Pro Laptop with 16 GB of Memory), my analysis was limited to a small representative sample of slices and mitochondria per dataset (~1%). In future work, embeddings can be computed across all slices across all mitochondria.

### Producing dense embeddings

Dense embeddings were computed as simple bilinear interpolation. This calculation distorts the boundaries of mitochondria because of the low-resolution patch sizes. A dense decoder can be trained per (Oquab, 2023) to improve the per-pixel dense embeddings calculation.

### Multiple Queries

Each dataset can be evaluated more comprehensively by incorporating multiple queries mitochondria in analysis. Two reasons for multiple queries can be defined.

First, we can select multiple mitochondrion and take a simple average across a dataset to create a meta-mitochondria, representative of the entire dataset. The same analyses described above can be applied. Each additional mitochondrion are independently retrieved then averaged into a single reference embeddings. Then, nearest and furthest neighbors are evaluated.

Second, multiple mitochondria can be analyzed to characterize the morphological diversity of the dataset. To characterize this diversity, we randomly select mitochondria and compute pairwise L2 distances between their embeddings across many random draws. The resulting distance distribution characterizes the dataset's morphological spread. 

The mean pairwise distance provides a diversity metric enabling direct quantitative comparison across datasets—a larger mean indicates greater heterogeneity.

### Proposal for Fine-tuning

We propose freezing pretrained DINOv3 weights and introducing Low-Rank Adaptation (LoRA) modules at selected attention layers. LoRA has demonstrated strong performance in adapting large vision transformers to downstream tasks with minimal trainable parameters, reducing compute and overfitting risk while preserving pretrained representations. Low rank selection and adapter placement will be guided by ablation experiments.

Training will be performed in two phase:

**Phase 1** – Self-Supervised Pretraining. The model will first undergo DINO-style self-supervised training on a combined labeled and unlabeled dataset. This phase leverages the full data distribution to learn robust, domain-specific features relevant to mitochondrial ultrastructure without requiring dense annotations.

**Phase 2** – Supervised Segmentation Fine-Tuning. Building on Phase 1 representations, we fine-tune on the labeled subset using a supervised segmentation objective. This two-stage curriculum exploits unlabeled data to regularize learned features before committing to annotation-dependent optimization, consistent with semi-supervised learning best practices.

**Evaluation** - Fine-tuning quality will be assessed quantitatively against held-out labeled data using standard segmentation metrics (IoU, Dice). Phase 1 will additionally be evaluated via linear probing prior to supervised fine-tuning.

# AI/LLM Usage statement

Claude and Claude Code were used throughout this project for research, idea development, code implementation, and text editing. I used Claude to identify relevant research papers and to develop ideas, with Claude pointing me toward useful background reading along the way.

For coding, I designed the high-level structure and workflow, while Claude assisted in scaffolding function implementations and migrating exploratory analyses from notebooks into classes.

# Bibliography

1. Dosovitskiy, Alexey, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, et al. 2020. “An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale.” arXiv [Cs.CV]. https://doi.org/10.48550/ARXIV.2010.11929.
2. Oquab, Maxime, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, et al. 2023. “DINOv2: Learning Robust Visual Features without Supervision.” arXiv [Cs.CV]. https://doi.org/10.48550/ARXIV.2304.07193.
3. González-Marfil, Aitor, Estibaliz Gómez-de-Mariscal, and Ignacio Arganda-Carreras. 2025. “DINOSim: Zero-Shot Object Detection and Semantic Segmentation on Microscopy Images.” bioRxiv. https://doi.org/10.1101/2025.03.09.642092.
4. Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. “LoRA: Low-Rank Adaptation of Large Language Models.” arXiv [Cs.CL]. arXiv. https://doi.org/10.48550/arXiv.2106.09685.
5. Ranftl, René, Alexey Bochkovskiy, and Vladlen Koltun. 2021. “Vision Transformers for Dense Prediction.” arXiv [Cs.CV]. https://doi.org/10.48550/ARXIV.2103.13413.
