#!/usr/bin/env python3
"""
Geocode texts in a Parquet file using the IRChel Geoparser.

Usage:
    python geocode_parquet.py input_file.parquet [--batch-size N] [--spacy-model MODEL] \
                                           [--transformer-model MODEL] [--gazetteer NAME]

The script reads `input_file.parquet`, expects a column named `text`, runs toponym recognition
and resolution in batches, and writes out `input_file_geocoded.parquet` including a new
`geocoded` column with JSON-serialized geoparsing results.
"""
import argparse
import os
import sys
import gc
# import pickle

import torch
import pandas as pd

# ensure src path is on sys.path if running from project root
PROJECT_ROOT: str = os.path.abspath(os.path.dirname("__file__"))
print(f'Project root: {PROJECT_ROOT}')
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.irchel_geoparser.geoparser import Geoparser  # noqa

# print available GPUs
if torch.cuda.is_available():
    print("CUDA GPUs detected:")
    for i in range(torch.cuda.device_count()):
        print(f"[{i}] {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA GPUs detected.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Geocode texts in a Parquet file using the IRChel Geoparser."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input Parquet file containing a 'text' column",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Number of texts to process per batch (default: 50000)",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="en_core_web_trf",
        help="spaCy model name (default: en_core_web_trf)",
    )
    parser.add_argument(
        "--transformer-model",
        type=str,
        default="dguzh/geo-all-distilroberta-v1",
        help="SentenceTransformer model name or path",
    )
    parser.add_argument(
        "--gazetteer",
        type=str,
        default="geonames",
        help="Gazetteer name to use (default: geonames)",
    )
    return parser.parse_args()


def geocode_texts(
    texts: list[str], geoparser: Geoparser, batch_size: int
) -> list:
    """
    Run geoparsing in batches, freeing GPU memory between batches.
    """
    # TODO: if GPU memory is low, doing thus with python multiprocessing might be better
    docs: list = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        docs_batch = geoparser.parse(batch)
        docs.extend(docs_batch)
        # free up GPU VRAM
        gc.collect()
        torch.cuda.empty_cache()
    return docs


def geodoc_to_dict(geodoc) -> dict:
    """
    Convert a GeoDoc to a JSON-serializable dict, extracting location details and score.
    """
    return {
        "text": getattr(geodoc, "text", None),
        "toponyms": [
            {
                "text": topo.text,
                "start_char": topo.start_char,
                "end_char": topo.end_char,
                # Extract resolved location details if available
                "loc_name": topo.location.get("name") if topo.location else None,
                "country_name": topo.location.get("country_name") if topo.location else None,
                "latitude": topo.location.get("latitude") if topo.location else None,
                "longitude": topo.location.get("longitude") if topo.location else None,
                "score": topo.score,
            }
            for topo in getattr(geodoc, "toponyms", [])
        ],
    }


def main():
    args = parse_args()
    input_path = args.input_file

    # read input parquet
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    if 'text' not in df.columns:
        print("Input file must contain a 'text' column.")
        sys.exit(1)

    # initialize geoparser
    geoparser = Geoparser(
        spacy_model=args.spacy_model,
        transformer_model=args.transformer_model,
        gazetteer=args.gazetteer,
    )

    texts = df['text'].astype(str).tolist()
    print(f"Geoparsing {len(texts)} documents in batches of {args.batch_size}...")

    docs = geocode_texts(texts, geoparser, args.batch_size)

    # serialize to JSON strings
    df['geocoded_dict'] = [geodoc_to_dict(d) for d in docs]

    # write output
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_geocoded.parquet"
    df.to_parquet(output_path)
    print(f"Saved geocoded DataFrame to: {output_path}")

    # unfortunately, GeoDoc objects are not serializable to python objects ...
    # with open(f"{base}_geodocs.pickle", 'wb') as file:
    # pickle.dump(docs, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
