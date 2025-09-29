#!/usr/bin/env python3
"""
ESDA pipeline module: performs topic modelling (with optimisation), emotion classification,
and disaster-relatedness classification.

Callable via:
    python -m esda_pipeline --input-file INPUT.parquet --output-path OUTPUT_DIR [--sample N]
"""
import os
import gc
import argparse
import torch
import pandas as pd
from nltk.corpus import stopwords
from tqdm.auto import tqdm

from src.nlp.esda_topic_modelling import TopicModellingOptimiser
from src.nlp.esda_classification import clean_bert, classify_emotions_multilabel, classify_disaster_relatedness


def main():
    parser = argparse.ArgumentParser(
        description="ESDA pipeline: topic modelling, emotion & disaster-relatedness classification"
    )
    parser.add_argument(
        "--input-file", "-i",
        required=True,
        help="Path to input Parquet file containing text data"
    )
    parser.add_argument(
        "--output-path", "-o",
        required=True,
        help="Directory in which to save the output Parquet files"
    )
    parser.add_argument(
        "--sample", "-s",
        type=int,
        default=None,
        help="If set, randomly sample this many rows for testing; otherwise use full data"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Load input data
    df = pd.read_parquet(args.input_file)
    if args.sample:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)

    # Ensure cleaned_text column
    if 'cleaned_text' not in df.columns:
        if 'text' in df.columns:
            tqdm.pandas(desc="Cleaning text")
            df['cleaned_text'] = df['text'].progress_apply(clean_bert)
        else:
            raise KeyError("Input data must contain either a 'cleaned_text' or 'text' column.")

    # Topic modelling
    topic_optimiser = TopicModellingOptimiser(
        embedding_model='intfloat/multilingual-e5-small',
        random_state=42,
        verbose=True
    )
    # Precompute the embeddings
    topic_optimiser.precompute_embeddings(df['cleaned_text'].tolist(), path=None)
    topic_optimiser.precompute_reduced_embeddings(path=None)

    topic_eval_df: pd.DataFrame = topic_optimiser.evaluate_topic_model(
        texts=df['cleaned_text'].tolist(),
        evaluation_texts=df['cleaned_text'].tolist(),  # using same data for simplicity; swap in held-out data if available
        params_nr_topics=[10, 20, 30],
        clustering_algorithms=['hdbscan', 'kmeans'],
        stop_words=(stopwords.words('english') + stopwords.words('spanish') + ['@user', 'http']),
        min_cluster_size=100,
        verbose=True
    )

    # extract the best configuration
    best_config: pd.Series = topic_eval_df.sort_values('train_quality', ascending=False).iloc[0]
    best_algo: str = best_config['clustering_algorithm']
    best_nr_topics: int = int(best_config['nr_topics'])
    print(f"Best config → algorithm: {best_algo}, nr_topics: {best_nr_topics}")

    # apply the best configuration
    topic_info_df, coherence, diversity, topic_ids = topic_optimiser.fit_transform(
        texts=df['cleaned_text'].tolist(),
        nr_topics=best_nr_topics,
        stop_words=None,
        clustering_algorithm=best_algo,
        min_cluster_size=100,
        verbose=False
    )

    # free up space
    del topic_optimiser
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    #  inspect outputs
    print("\nTopic information:")
    print(topic_info_df)

    # add topic IDs to the original dataframe
    topic_info_df.set_index('Topic', inplace=True)
    df['topic_id'] = topic_ids
    df["keywords"] = df["topic_id"].apply(lambda x: topic_info_df.loc[x, "Representation"])
    df["topic_label"] = df["keywords"].apply(lambda x: ', '.join(x))

    # Emotion classification
    emotions: pd.DataFrame = classify_emotions_multilabel(texts=df['cleaned_text'].tolist())
    df.loc[:, emotions.columns] = emotions.to_numpy()

    # Disaster-relatedness classification
    df['disaster_related'] = classify_disaster_relatedness(texts=df['cleaned_text'].tolist())

    # Save ESDA-augmented data
    input_stem = os.path.splitext(os.path.basename(args.input_file))[0]
    out_file = os.path.join(args.output_path, f"{input_stem}_esda.parquet")
    print(f"Saving enriched data to {out_file}")
    df.to_parquet(out_file, index=False)

    # Save topic information separately
    topics_file = os.path.join(args.output_path, f"{input_stem}_topics.parquet")
    print(f"Saving topic info to {topics_file}")
    topic_info_df.to_parquet(topics_file, index=False)


if __name__ == "__main__":
    main()
