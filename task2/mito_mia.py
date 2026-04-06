"""
mito_mia.py — Mitochondria morphology and image analysis pipeline.

Runs the full single-dataset workflow:
  1. Load EM + segmentation data
  2. Generate patches across the volume
  3. Build the mito catalog (random subset of slices)
  4. Compute patch embeddings
  5. Compute per-mito reference vectors
  6. Run reference analysis (distances, closest/furthest plots)

All figures are saved to output_dir.
"""

import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving without display
import matplotlib.pyplot as plt

from src.model import load_vits16_model,load_vits16_model_hf
from src.dataset_pipeline import DatasetPipeline
from src.visualizer import Visualizer
from src.reference_analyzer import ReferenceAnalyzer
import os
from src.timer import Timer


def run(output_dir: str = "./output/"):
    """Run the full single-dataset mito analysis pipeline.

    Parameters:
        output_dir: Directory where all figures are saved. Assumed to exist.
    """

    # Make the output directory
    timer = Timer()
    os.makedirs(output_dir, exist_ok=True)

    # Configuration for number of random samples
    num_random_samples = 10

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("Loading model...")
    model = load_vits16_model_hf()

    # ------------------------------------------------------------------
    # 2. Run dataset pipeline 1
    # ------------------------------------------------------------------

    group_url = 's3://janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.n5/em/fibsem-uint8/' 
    seg_url = 's3://janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.n5/labels/mito_seg/'
    pipeline_mus_liver = DatasetPipeline(group_url, seg_url, 's1', 's0', model, name='mus-liver',
        output_dir=output_dir, num_random_samples=num_random_samples)
    pipeline_mus_liver.run()

    # ------------------------------------------------------------------
    # 3. Run dataset pipeline 2
    # ------------------------------------------------------------------

    group_url = 's3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.n5/em/fibsem-uint8/' 
    seg_url = 's3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.n5/labels/empanada-mito_seg/'
    pipeline_mus_kidney = DatasetPipeline(group_url, seg_url, 's1', 's0', model, name='mus-kidney',
        output_dir=output_dir, num_random_samples=num_random_samples)
    pipeline_mus_kidney.run()

    # ------------------------------------------------------------------
    # 4. Run comparison pipeline
    # ------------------------------------------------------------------
    
    # Analyse the second data set with the first's mitochondrian selected.

    from src.cross_dataset_analyzer import CrossDatasetAnalyzer

    cross = CrossDatasetAnalyzer(pipeline_mus_liver, pipeline_mus_kidney)                                                                                    
    cross.run()

    timer.print_time("Total runtime")

if __name__ == "__main__":
    run()
