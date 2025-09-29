"""Pre-defined functions for emotion, sentiment and disaster-relatedness classification.
"""
import re
import gc
import html
import torch
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset


def clean_bert(x: str) -> str:
    """Clean texts for transformer-based processing

    Args:
        x (str): The input string.

    Returns:
        str: The processed string.
    """
    x = re.sub(r"@\w+", "@user", x)  # normalise @references
    x = re.sub(r'https?://\S+|www\.\S+', 'http', x)  # normalise links
    x = x.replace("\n", "")  # remove extra white space
    x = html.unescape(x)  # remove html entities
    return x


def get_class_probs(texts: list[str], model: str = 'MilaNLProc/xlm-emo-t',
                    batch_size: int = 16, device: str = 'cuda') -> pd.DataFrame:
    """Get classification probabilities for a list of texts using a specified model.

    Args:
        texts (list[str]): A list of texts to classify.
        model (str, optional): The model to use for classification. Defaults to 'MilaNLProc/xlm-emo-t'.
        batch_size (int, optional): The batch size for processing. Defaults to 16.
        device (str, optional): The device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        pd.DataFrame: A DataFrame containing the classification probabilities for each text.
    """
    # create a HuggingFace dataset for efficient inference
    dataset: Dataset = Dataset.from_dict({'text': texts})

    # and initialise a simple pipeline
    classifier = pipeline(model=model, device=device, truncation=True, max_length=512,
                          batch_size=batch_size, return_all_scores=True)

    # start classifying
    result_dictlist: list[dict] = []
    for out in tqdm(classifier(KeyDataset(dataset, 'text')), total=len(dataset)):
        # extract all scores
        result_dict: dict = {}
        for entry in out:
            result_dict[f"p_{entry['label']}"] = entry['score']
        result_dictlist.append(result_dict)

    # free up space
    del classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame.from_dict(result_dictlist)


def classify_emotions_multilabel(texts: list[str], model: str = 'MilaNLProc/xlm-emo-t', min_score: float = 0.5,
                                 batch_size: int = 16, device: str = 'cuda'):
    """Classify emotions in a list of texts using a specified model.

    Args:
        texts (list[str]): A list of texts to classify.
        model (str, optional): The model to use for classification. Defaults to 'MilaNLProc/xlm-emo-t'.
        min_score (float, optional): The minimum score threshold for classification. Defaults to 0.5.
        batch_size (int, optional): The batch size for processing. Defaults to 16.
        device (str, optional): The device to use for computation ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        pd.DataFrame: A DataFrame with the classification results. Each column represents an emotion,
                      and each row represents a text. The values are boolean, indicating whether the
                      emotion is present in the text.
    """
    # extract probabilities for each class
    prob_df: pd.DataFrame = get_class_probs(texts=texts, model=model, batch_size=batch_size,
                                            device=device)

    # iterate over all result columns and make a classification
    labels: list[str] = []
    for column in prob_df.columns:
        label: str = column[2:]
        prob_df[label] = prob_df[column] > min_score
        labels.append(label)

    # if a document has no classification at all, then make it no_emotion
    prob_df['no_emotion'] = (prob_df[labels] == False).all(axis=1)  # noqa
    return prob_df


def classify_sentiments(texts: list[str], model: str = 'cardiffnlp/twitter-xlm-roberta-base-sentiment',
                        batch_size: int = 16, device: str = 'cuda') -> pd.DataFrame:
    """Get classification probabilities and a single sentiment label for a list of texts using a specified model.

    Args:
        texts (list[str]): A list of texts to classify.
        model (str, optional): The model to use for classification. Defaults to 'cardiffnlp/twitter-xlm-roberta-base-sentiment'.
        batch_size (int, optional): The batch size for processing. Defaults to 16.
        device (str, optional): The device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        pd.DataFrame: A DataFrame containing the classification probabilities for each text.
    """
    # create a HuggingFace dataset for efficient inference
    dataset: Dataset = Dataset.from_dict({'text': texts})

    # and initialise a simple pipeline
    classifier = pipeline(model=model, device=device, truncation=True, max_length=512,
                          batch_size=batch_size, return_all_scores=True)

    # start classifying
    result_dictlist: list[dict] = []
    for out in tqdm(classifier(KeyDataset(dataset, 'text')), total=len(dataset)):
        # extract all scores
        result_dict: dict = {}
        for entry in out:
            result_dict[f"p_{entry['label']}"] = entry['score']

        # extract maximum label
        max_entry: dict = max(out, key=lambda x: x['score'])
        result_dict['sentiment'] = max_entry['label']

        result_dictlist.append(result_dict)

    return pd.DataFrame.from_dict(result_dictlist)


def classify_disaster_relatedness(texts: list[str], model_name: str = 'hannybal/disaster-twitter-xlm-roberta-al',
                                  tokenizer_name: str = 'cardiffnlp/twitter-xlm-roberta-base', batch_size: int = 8,
                                  max_length: int = 512, device: str = 'cuda') -> list[int]:
    """
    Classify a list of texts as disaster-related or not.

    Args:
        texts (List[str]): List of preprocessed text strings.
        model_name (str): HuggingFace model identifier for classification.
        tokenizer_name (str): HuggingFace tokenizer identifier.
        batch_size (int): Number of samples per batch through the pipeline.
        max_length (int): Max token length for truncation/padding.

    Returns:
        List[int]: Binary labels (1 = related, 0 = not related) for each input text.
    """
    if not texts:
        return []

    pipe = pipeline(
        'text-classification',
        model=model_name,
        tokenizer=tokenizer_name,
        add_special_tokens=True,
        padding='max_length',
        truncation='longest_first',
        max_length=max_length,
        device=device,
        batch_size=batch_size,
        return_all_scores=True,
    )

    dataset = Dataset.from_dict({'text': texts})
    labels: list[int] = []

    for output in tqdm(pipe(KeyDataset(dataset, 'text')), total=len(texts), desc="Disaster classification"):
        # output is a list of dicts like [{"label": "LABEL_0", "score": 0.9}, ...]
        best = max(output, key=lambda x: x['score'])
        labels.append(1 if best['label'] == 'LABEL_1' else 0)

    # free up space
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return labels
