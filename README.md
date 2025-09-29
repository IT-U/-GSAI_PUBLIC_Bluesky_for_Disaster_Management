# Bluesky as a Social Media Data Source for Disaster Management 
This repository contains the replication code and materials for the study:

**Hanny, D., Schmidt, S. & Resch, B. (2025).**
*Bluesky as a Social Media Data Source for Disaster Management: Investigating Spatio-temporal, Semantic and Emotional Patterns for Floods and Wildfires.*
[Submitted to the Journal of *Computational Social Science*]

## 📄 Overview

This study evaluates Bluesky as an alternative social media data source for geographically grounded disaster management. Our analysis pipeline includes:

1. **Data collection**: Custom keyword-based crawling of Bluesky posts via API
2. **Geoparsing**: Large-scale geoparsing of location mentions in disaster-related content
3. **Content analysis**:
   - Semantic classification of disaster-related posts
   - Emotion detection using multilingual NLP models
4. **Spatio-temporal analysis**:
   - Geographic distribution of location mentions
   - Temporal patterns in posting activity
5. **Comparative analysis**: Case studies of the September 2024 Central Europe floods and January 2025 Southern California wildfires

## 📁 Repository Structure
The analysis pipeline is spread across several scripts and Jupyter notebooks. A full overview is available below.

```
├── environment.yml # Conda environment file

├── notebooks/      # Data exploration, (spatial) analysis and visualisation
│ ├── bksy_data_collection.ipynb    # Data collection incl. geoparsing
│ ├── bsky_europe_floods.ipynb      # Analysis of 2024 Central Europe floods
│ ├── bsky_socal_wildfires.ipynb    # Analysis of 2025 Southern California wildfires
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
```

### 🤖 Bluesky Crawling Algorithm
One key contribution is our iterative crawling algorithm which uses the `app.bsky.feed.searchPosts` API endpoint and circumvents its currently disfunctional cursor parameter (as of September 2025).  It uses a list of keywords and a time interval as interval. Then, for each keyword, the algorithm iterates over the time interval in steps of a pre-defined size (e.g. 30 minutes).

The implemented algorithm is available in `scripts/crawling/*`.

### ⌨️ NLP Analysis
Additionally, our NLP analysis pipeline, involving semantic disaster-relatedness classification and emotion classfication is available as a ready-to-use script.

```bash
python -m scripts.esda.nlp_analysis --input-file "<input-file>" --output-path "<output-file>"
```

## ⚙️ Getting Started
To replicate the experiments, create the conda environment:

```bash
conda env create -f environment.yml
conda activate bsky-disaster
```

Then, run the desired notebooks or scripts.

## 📊 Data Availability

Due to Twitter’s (now X) API terms, we are unable to share full tweet content. However, we provide tweet IDs, ground truth relevance labels and our engineered spatiotemporal features. These can be accessed and rehydrated using the X v2 API. This dataset and accompanying metadata are available on Harvard Dataverse.

## 📖 Citation

If you use this code or dataset in your research, please cite our work accordingly.

@article{Hanny.2025,
  title     = {Bluesky as a Social Media Data Source for Disaster Management: Investigating Spatio-temporal, Semantic and Emotional Patterns for Floods and Wildfires},
  author    = {Hanny, David and Schmidt, Sebastian and Resch, Bernd},
  journal   = {TBD},
  year      = {2025}
}