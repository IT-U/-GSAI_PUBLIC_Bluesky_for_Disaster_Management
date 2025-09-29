# Bluesky as a Social Media Data Source for Disaster Management 
This repository contains the replication code and materials for the study:

Hanny, D., Schmidt, S. & Resch, B. (2025).
Bluesky as a Social Media Data Source for Disaster Management: Investigating Spatio-temporal, Semantic and Emotional Patterns for Floods and Wildfires.
[Submitted to the Journal of Computational Social Science]

📄 Overview

This study presents a multimodal relevance classification approach that integrates textual, spatial, and temporal features to improve relevance classification of social media posts in natural disaster scenarios. The pipeline includes:

    Pre-processing of geo-referenced tweets
    Feature engineering and evaluation (spatial, temporal, co-occurrence)
    Non-text classifier training evaluation
    Text classifier (TwHIN-BERT) training and evaluation
    Multimodal fusion with feature concatenation, partial stacking and in-context learning
    Model comparison, visualisation and explanation with SHAP

📁 Repository Structure

├── environment.yml # Conda environment file

├── notebooks/      # Data exploration, (spatial) analysis and visualisation
│ ├── bsky_europe_floods.ipynb      # 2024 Central Europe floods
│ ├── bsky_socal_wildfires.ipynb    # 2025 Southern California wildfires
│ ├── ground_reference_prep.ipynb   # Ground reference satellite data
│ ├── ...
│ ├── h3_aggregation_bluesky.ipynb  # Spatial analysis part 1
│ ├── overlap_ground_truth.ipynb    # Spatial analysis part 2
│ └── visualisation.R  # Data visualisation

├── scripts/  # Non-interactive scripts for data collection and processing
│ ├── crawling/   # Iterative crawling for our two use cases
│ │ ├── crawl_bsky_europe_floods.py
│ │ ├── crawl_bsky_socal_wildfires.py
│ ├── esda/  # Data processing
│ │ ├── nlp_analysis.py  # Disaster-relatedness classification, emotion classification, exploratory topic modelling

├── src/ # Helper modules and reusable functions
│ ├── bsky_search.py  # Bluesky crawling algorithm
│ ├── nlp/  # Functions for text processing
│ │ ├── esda_classification.py  # Semantic and emotion classification
│ │ └── ... 
│ ├── nlp/  # Functions for text processing
│ │ ├── ...
└── README.md # Project documentation

Each notebook corresponds to a specific step in the pipeline, from data loading and preprocessing to model training, inference, and evaluation. Reusable functions are available in the src/ directory.
⚙️ Getting Started
1. Create environment

To replicate the experiments, create the conda environment:

conda env create -f environment.yml
conda activate relevance-classification

2. Run notebooks

Execute the numbered notebooks (notebooks/01_... to 09_...) in order. Each notebook is self-contained and documented.
📊 Data Availability

Due to Twitter’s (now X) API terms, we are unable to share full tweet content. However, we provide tweet IDs, ground truth relevance labels and our engineered spatiotemporal features. These can be accessed and rehydrated using the X v2 API. This dataset and accompanying metadata are available on Harvard Dataverse.
📖 Citation

If you use this code or dataset in your research, please cite our work accordingly.

@article{hanny2025multimodalRelevance,
  title     = {A Multimodal GeoAI Approach to Combining Text with Spatiotemporal Features for Enhanced Relevance Classification of Social Media Posts in Disaster Response},
  author    = {Hanny, David and Schmidt, Sebastian and Gandhi, Shaily and Granitzer, Michael and Resch, Bernd},
  journal   = {TBD},
  year      = {2025}
}

🛠 Contact
In case of questions, please contact: David Hanny (david.hanny@it-u.at), IT:U Interdisciplinary Transformation University Austria
