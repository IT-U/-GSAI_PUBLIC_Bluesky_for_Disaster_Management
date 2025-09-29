"""Topic modelling for smart people - a framework.
"""
import concurrent.futures
import openai
import nltk
import numpy as np
import pandas as pd
from typing import Tuple
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from nltk.corpus import stopwords
from bertopic.representation import OpenAI, TextGeneration
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from src.nlp.metrics import compute_avg_topic_npmi, compute_topic_diversity

# download stopwords just in case
nltk.download('stopwords')


class TopicModellingOptimiser:
    """A class for optimal topic modelling based on given input texts.
    """

    def __init__(self, embedding_model: str = 'Alibaba-NLP/gte-multilingual-base',
                 random_state: int = 42, verbose: bool = True) -> None:
        self.embedding_model: str = embedding_model
        self.random_state: int = random_state
        self.verbose: bool = verbose
        self.embeddings: np.ndarray = None
        self.reduced_embeddings: np.ndarray = None

    def precompute_embeddings(self, texts: list[str], path: str | None) -> None:
        """
        Precomputes embeddings for a given list of texts using a specified SentenceTransformer model.

        Args:
            texts (list[str]): A list of text samples for which embeddings need to be computed.
            path (str | None): A file path where the embeddings will be saved as a `.npy` file.
                If `None`, the embeddings will not be saved to disk.

        Returns:
            None: The embeddings are stored in the `self.embeddings` attribute for later use.
            If a `path` is provided, the embeddings are also saved as a `.npy` file at the specified location.
        """
        sentence_model: SentenceTransformer = SentenceTransformer(self.embedding_model)
        self.embeddings: np.ndarray = sentence_model.encode(texts, show_progress_bar=self.verbose)

        if path is not None:
            np.save(path, self.embeddings)

    def load_embeddings(self, path: str) -> None:
        """Loads saved embeddings from a secified path.

        Args:
            path (str): A file path where the embeddings array is saved as a `.npy` file.
        """
        self.embeddings: np.ndarray = np.load(path)

    def precompute_reduced_embeddings(self, path: str | None) -> None:
        """
        Reduces the dimensionality of precomputed embeddings using UMAP and optionally saves the result to disk.

        Args:
            path (str | None): A file path where the reduced embeddings will be saved as a `.npy` file.
                If `None`, the reduced embeddings will not be saved.

        Raises:
            ValueError: If full embeddings (`self.embeddings`) have not been computed or loaded before calling this method.

        Returns:
            None: The reduced embeddings are stored in the `self.reduced_embeddings` attribute for later use.
            If a `path` is provided, the reduced embeddings are also saved as a `.npy` file at the specified location.
        """
        if self.embeddings is None:
            raise ValueError('Full embeddings have not been computed or loaded.')

        # create and apply UMAP
        umap_model: UMAP = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, random_state=self.random_state,
                                verbose=True, metric='cosine')
        self.reduced_embeddings: np.ndarray = umap_model.fit_transform(self.embeddings)

        if path is not None:
            np.save(path, self.reduced_embeddings)

    def load_reduced_embeddings(self, path: str) -> None:
        """Loads reduced embeddings from a secified path.

        Args:
            path (str): A file path where the reduced embeddings `.npy` file is located.
        """
        self.reduced_embeddings: np.ndarray = np.load(path)

    def fit_transform(self, texts: list[str],
                      nr_topics: int | str = 'auto',
                      stop_words: list[str] = stopwords.words('english'),
                      clustering_algorithm: str = 'hdbscan',
                      min_cluster_size: int = 100,
                      zeroshot_topic_list: list[str] | None = None,
                      zeroshot_min_similarity: float = 0.5,
                      verbose: bool = True,
                      openai_key: str | None = None,
                      huggingface_model: str | None = None,
                      embeddings: np.ndarray | None = None,
                      reduced_embeddings: np.ndarray | None = None
                      ) -> Tuple[pd.DataFrame, float, float, list[int]]:
        """
        Performs topic modeling on a set of text documents, extracting topics and their quality metrics.

        Args:
            texts (list[str]): A list of text samples for topic modeling.
            nr_topics (int | str, optional): Number of topics to generate. Defaults to 'auto'.
                If using k-means clustering, this sets the number of clusters explicitly.
            stop_words (list[str], optional): List of stop words to exclude from the topic vectorization process.
                Defaults to an English stopwords list.
            clustering_algorithm (str, optional): Clustering algorithm to use. Supported options:
                'hdbscan' (default) or 'kmeans'.
            min_cluster_size (int, optional): Minimum cluster size for the HDBSCAN clustering algorithm.
                Ignored if using k-means. Defaults to 100.
            zeroshot_topic_list (list[str] | None, optional): A predefined list of zero-shot topics for guidance.
                If `None`, topics are automatically inferred. Defaults to `None`.
            zeroshot_min_similarity (float, optional): Minimum cosine similarity threshold for assigning zero-shot topics.
                Defaults to 0.5.
            verbose (bool, optional): Whether to display progress and debug messages. Defaults to `True`.
            openai_key (str | None, optional): API key for using OpenAI models to generate topic labels.
                If `None`, OpenAI models are not used. Defaults to `None`.
            huggingface_model (str | None, optional): HuggingFace model to use for topic label generation.
                If `None`, HuggingFace models are not used. Defaults to `None`.
            embeddings (np.ndarray | None, optional): Pre-generated embeddings if available.
                If `None`, the class instance embeddings will be used.
            reduced_embeddings (np.ndarray | None, optional): Pre-generated embeddings if available.
                If `None`, the class instance embeddings will be used.

        Returns:
            Tuple[pd.DataFrame, float, float, list[int]]:
                - A DataFrame with topic representations and their details.
                - The coherence score (float) of the topics, measuring semantic similarity between keywords.
                - The diversity score (float) of the topics, assessing uniqueness across topics.
                - The list of topic classifications for the input texts.

        Raises:
            ValueError: If embeddings or reduced embeddings have not been precomputed.
            ValueError: If the specified clustering algorithm is not implemented.

        Notes:
            - The method uses UMAP for dimensionality reduction and BERTopic for topic modeling.
            - Topic quality is evaluated using coherence and diversity metrics.
        """
        if (self.embeddings is None or self.reduced_embeddings is None) and (embeddings is None and reduced_embeddings is None):
            raise ValueError('Embeddings have not yet been computed and are not provided. Please do that first.')

        # in case embeddings are provided, use those, otherwise use the precomputed ones
        if embeddings is None:
            embeddings: np.ndarray = self.embeddings
        if reduced_embeddings is None:
            reduced_embeddings: np.ndarray = self.reduced_embeddings

        #########################
        #       prep work       #
        #########################
        # custom vectoriser to minimise stopwords
        vectoriser_model: CountVectorizer = CountVectorizer(stop_words=stop_words)

        # initialise clustering technique
        if clustering_algorithm == 'hdbscan':
            cluster_model = HDBSCAN(min_cluster_size=min_cluster_size,
                                    metric='euclidean',
                                    cluster_selection_method='eom',
                                    prediction_data=True)
        elif clustering_algorithm == 'kmeans':
            cluster_model = KMeans(n_clusters=nr_topics, random_state=self.random_state)
            nr_topics = None  # override nr_topics
        else:
            raise ValueError('Clustering algorithm not implemented.')

        # pre-defined representation prompts
        prompt: str = """
        I have a topic that contains the following documents:
        [DOCUMENTS]
        The topic is described by the following keywords: [KEYWORDS]

        Based on the information above, extract a short but highly descriptive topic label of at most 5 words.
        Make sure you to only return the label and nothing more.
        """

        example_prompt: str = """
        I have a topic that contains the following documents:
        - Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
        - Meat, but especially beef, is the word food in terms of emissions.
        - Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

        The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

        Based on the information above, extract a short but highly descriptive topic label of at most 5 words.
        Make sure you to only return the label and nothing more.
        """ # noqa
        example_answer: str = 'Environmental impacts of eating meat'

        # define representation
        representation_model: dict = {}
        if openai_key is not None:
            client = openai.OpenAI(api_key=openai_key)
            representation_model['openai'] = OpenAI(client, model="gpt-4o-mini",
                                                    delay_in_seconds=10, chat=True,
                                                    nr_docs=min_cluster_size,
                                                    prompt=prompt)

        # TODO: fix this
        if huggingface_model is not None:
            # recommended: "microsoft/Phi-3.5-mini-instruct"
            generation_model = AutoModelForCausalLM.from_pretrained(
                huggingface_model,
                device_map='cuda',
                torch_dtype='auto',
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(huggingface_model)

            # define a system prompt with examples
            messages = [
                {'role': 'system', 'content': 'You are a helpful, respectful and honest assistant for labeling topics.'},
                {'role': 'user', "content": example_prompt},
                {'role': 'assistant', 'content': example_answer},
                {'role': 'user', 'content': prompt},
            ]

            generator = pipeline('text-generation', model=generation_model, tokenizer=tokenizer,
                                 max_new_tokens=50, repetition_penalty=1.1)
            representation_model['huggingface'] = TextGeneration(generator, prompt=messages)

        # define precomputed UMAP instance
        precomputed_umap_model = PrecomputedUMAP(embeddings=embeddings,
                                                 reduced_embeddings=reduced_embeddings)

        #########################
        # build the topic model #
        #########################
        # actual topic model application

        # Option 1: We have zero shot topics
        if zeroshot_topic_list is not None:
            self.topic_model: BERTopic = BERTopic(
                umap_model=precomputed_umap_model,
                hdbscan_model=cluster_model,
                zeroshot_topic_list=zeroshot_topic_list,
                calculate_probabilities=False,
                zeroshot_min_similarity=zeroshot_min_similarity,
                embedding_model=self.embedding_model,
                vectorizer_model=vectoriser_model,
                nr_topics=nr_topics,
                verbose=verbose,
                representation_model=representation_model)
        # Option 2: We do not have any zero shot topics
        else:
            self.topic_model: BERTopic = BERTopic(
                umap_model=precomputed_umap_model,
                hdbscan_model=cluster_model,
                zeroshot_topic_list=zeroshot_topic_list,
                calculate_probabilities=False,
                zeroshot_min_similarity=zeroshot_min_similarity,
                embedding_model=self.embedding_model,
                vectorizer_model=vectoriser_model,
                nr_topics=nr_topics,
                verbose=verbose,
                representation_model=representation_model)

        # extract topics
        topics, _ = self.topic_model.fit_transform(texts, embeddings=embeddings)
        self.topic_representation: pd.DataFrame = self.topic_model.get_topic_info()

        # lastly, we can evaluate topic quality
        self.topic_keywords: list[list[str]] = self.topic_representation['Representation'].tolist()
        self.coherence: float = compute_avg_topic_npmi(documents=texts, topics=self.topic_keywords,
                                                       use_ppmi=True, verbose=verbose)
        self.diversity: float = compute_topic_diversity(topics=self.topic_keywords)
        return self.topic_representation, self.coherence, self.diversity, topics

    def evaluate_topic_model(
            self,
            texts: list[str],
            evaluation_texts: list[str],
            params_nr_topics: list[int],
            clustering_algorithms: list[str] = ['hdbscan', 'kmeans'],
            stop_words: list[str] = stopwords.words('english'),
            min_cluster_size: int = 100,
            zeroshot_topic_list: list[str] | None = None,
            zeroshot_min_similarity: float = 0.5,
            verbose: bool = True
            ) -> pd.DataFrame:
        """
        Evaluates topic modeling performance across various clustering algorithms and topic numbers.

        Args:
            texts (list[str]): The training dataset of text samples for topic modeling.
            evaluation_texts (list[str]): The test dataset used to evaluate topic quality.
            params_nr_topics (list[int]): A list of numbers representing the potential topic counts to evaluate.
            clustering_algorithms (list[str], optional): A list of clustering algorithms to test.
                Supported options are 'hdbscan' and 'kmeans'. Defaults to ['hdbscan', 'kmeans'].
            stop_words (list[str], optional): List of stop words to exclude during topic vectorization.
                Defaults to `stopwords.words('english')`.
            min_cluster_size (int, optional): Minimum cluster size for the HDBSCAN clustering algorithm.
                Ignored for k-means. Defaults to 100.
            zeroshot_topic_list (list[str] | None, optional): A predefined list of zero-shot topics for clustering guidance.
                If `None`, topics are inferred. Defaults to `None`.
            zeroshot_min_similarity (float, optional): Minimum cosine similarity threshold for assigning zero-shot topics.
                Defaults to 0.5.
            verbose (bool, optional): Whether to display progress and debug messages. Defaults to `True`.

        Returns:
            pd.DataFrame: A DataFrame containing evaluation results for each clustering algorithm and
                topic number. The columns include:
                - 'clustering_algorithm': The clustering algorithm used ('hdbscan' or 'kmeans').
                - 'nr_topics': The number of topics specified for modeling ('auto' if inferred automatically).
                - 'actual_topics': The actual number of topics generated.
                - 'coherence': The coherence score (average NPMI) on the evaluation dataset.
                - 'diversity': The diversity score of the topics.
                - 'quality': A combined quality metric (`coherence * coherence`).
                - 'train_size': The number of samples in the training dataset.
                - 'test_size': The number of samples in the evaluation dataset.

        Raises:
            ValueError: If any input parameter is invalid or unsupported.

        Notes:
            - This method loops over combinations of clustering algorithms and topic counts.
            - Topic quality is assessed on test data using coherence (NPMI) and diversity metrics.
            - The evaluation results provide insights into the performance of different clustering techniques.
        """
        result_dictlist: list[dict] = []

        for clustering_algorithm in clustering_algorithms:
            print('----------------------')
            print(f'Evaluating {clustering_algorithm} for clustering!')
            # and over all nr_topic parameters
            for nr_topics in params_nr_topics:
                if clustering_algorithm == 'hdbscan' or nr_topics != 'auto':
                    print(f'Running topic model with nr_topics={nr_topics}')
                    topics, train_tc, train_td, _ = self.fit_transform(
                        texts=texts,
                        nr_topics=nr_topics,
                        stop_words=stop_words,
                        clustering_algorithm=clustering_algorithm,
                        min_cluster_size=min_cluster_size,
                        zeroshot_topic_list=zeroshot_topic_list,
                        zeroshot_min_similarity=zeroshot_min_similarity,
                        verbose=verbose,
                        embeddings=self.embeddings,
                        reduced_embeddings=self.reduced_embeddings
                    )

                    # evaluate topic quality on the test data set
                    topic_keywords: list[list[str]] = topics['Representation'].tolist()
                    test_tc: float = compute_avg_topic_npmi(documents=evaluation_texts,
                                                            topics=topic_keywords,
                                                            use_ppmi=True,
                                                            verbose=verbose)
                    test_td: float = compute_topic_diversity(topics=topic_keywords)

                    result_dictlist.append({
                        'clustering_algorithm': clustering_algorithm,
                        'nr_topics': nr_topics,
                        'actual_topics': topics.shape[0],
                        'train_size': len(texts),
                        'test_size': len(evaluation_texts),
                        'test_coherence': test_tc,
                        'test_diversity': test_td,
                        'test_quality': test_tc * test_td,
                        'train_coherence': train_tc,
                        'train_diversity': train_td,
                        'train_quality': train_tc * train_td
                    })
        return pd.DataFrame.from_dict(result_dictlist)

    def parallel_topic_evaluation(
            self,
            texts: list[str],
            evaluation_texts: list[str],
            params_nr_topics: list[int],
            clustering_algorithms: list[str] = ['hdbscan', 'kmeans'],
            stop_words: list[str] = stopwords.words('english'),
            min_cluster_size: int = 100,
            zeroshot_topic_list: list[str] | None = None,
            zeroshot_min_similarity: float = 0.5,
            verbose: bool = True,
            max_workers: int | None = None) -> pd.DataFrame:
        """
        Perform parallel evaluation of full topic modelling.

        Args:
            texts (list[str]): The training dataset of text samples for topic modeling.
            evaluation_texts (list[str]): The test dataset used to evaluate topic quality.
            params_nr_topics (list[int]): A list of numbers representing the potential topic counts to evaluate.
            clustering_algorithms (list[str], optional): A list of clustering algorithms to test.
                Supported options are 'hdbscan' and 'kmeans'. Defaults to ['hdbscan', 'kmeans'].
            stop_words (list[str], optional): List of stop words to exclude during topic vectorization.
                Defaults to `stopwords.words('english')`.
            min_cluster_size (int, optional): Minimum cluster size for the HDBSCAN clustering algorithm.
                Ignored for k-means. Defaults to 100.
            zeroshot_topic_list (list[str] | None, optional): A predefined list of zero-shot topics for clustering guidance.
                If `None`, topics are inferred. Defaults to `None`.
            zeroshot_min_similarity (float, optional): Minimum cosine similarity threshold for assigning zero-shot topics.
                Defaults to 0.5.
            verbose (bool, optional): Whether to display progress and debug messages. Defaults to `True`.
            max_workers (int | None, optional): The maximum number of worker processes to use
                for parallel execution. Defaults to `None`, which allows the executor to choose
                the number of workers based on the available system resources.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the results of the evaluations.
        """
        results = []

        # Run evaluations within individual processes
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks to execute for each sample size and topic number
            futures_to_params_dict: dict = {}  # store the associations between futures and parameter configurations
            for clustering_algorithm in clustering_algorithms:
                for nr_topics in params_nr_topics:
                    futures_to_params_dict[executor.submit(self.evaluate_topic_model,
                                                           texts=texts,
                                                           evaluation_texts=evaluation_texts,
                                                           params_nr_topics=[nr_topics],
                                                           clustering_algorithms=[clustering_algorithm],
                                                           stop_words=stop_words,
                                                           min_cluster_size=min_cluster_size,
                                                           zeroshot_topic_list=zeroshot_topic_list,
                                                           zeroshot_min_similarity=zeroshot_min_similarity,
                                                           verbose=verbose)] = (nr_topics, clustering_algorithm)

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures_to_params_dict):
                nr_topics, clustering_algorithm = futures_to_params_dict[future]  # read the parameters from the completed task
                try:
                    result = future.result()  # Get the result
                    results.append(result)
                except Exception as e:
                    print(f"Configuration ({nr_topics}, {clustering_algorithm}) failed with exception: {e}")

        # Combine all results into a single DataFrame (if applicable)
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no results

    def evaluate_sampled_topic_model(
            self,
            texts: list[str],
            evaluation_texts: list[str],
            params_nr_topics: list[int],
            sample_sizes: list[int],
            clustering_algorithms: list[str] = ['hdbscan', 'kmeans'],
            stop_words: list[str] = stopwords.words('english'),
            min_cluster_size: int = 100,
            zeroshot_topic_list: list[str] | None = None,
            zeroshot_min_similarity: float = 0.5,
            verbose: bool = True) -> pd.DataFrame:
        """
        Evaluate topic models across multiple sample sizes, clustering algorithms, and topic configurations.

        Args:
            texts (list[str]): The training dataset of text samples for topic modeling.
            evaluation_texts (list[str]): The test dataset used to evaluate topic quality.
            params_nr_topics (list[int]): A list of numbers representing the potential topic counts to evaluate.
            sample_sizes (list[int]): List of sample sizes to draw from the training texts.
            clustering_algorithms (list[str], optional): A list of clustering algorithms to test.
                Supported options are 'hdbscan' and 'kmeans'. Defaults to ['hdbscan', 'kmeans'].
            stop_words (list[str], optional): List of stop words to exclude during topic vectorization.
                Defaults to `stopwords.words('english')`.
            min_cluster_size (int, optional): Minimum cluster size for the HDBSCAN clustering algorithm.
                Ignored for k-means. Defaults to 100.
            zeroshot_topic_list (list[str] | None, optional): A predefined list of zero-shot topics for clustering guidance.
                If `None`, topics are inferred. Defaults to `None`.
            zeroshot_min_similarity (float, optional): Minimum cosine similarity threshold for assigning zero-shot topics.
                Defaults to 0.5.
            verbose (bool, optional): Whether to display progress and debug messages. Defaults to `True`.

        Returns:
            pd.DataFrame: A DataFrame with evaluation results, including metrics for training and test coherence,
                        topic diversity, and overall quality for each configuration.

        Notes:
            - Evaluates topic models with different sample sizes, clustering methods, and topic counts.
            - Results include metrics such as test coherence, topic diversity, and topic quality for both training and test sets.

        Example:
            results = topic_optimiser.evaluate_sampled_topic_model(
                texts=train_texts,
                evaluation_texts=test_texts,
                params_nr_topics=[10, 20, 30],
                sample_sizes=[1000, 5000],
                clustering_algorithms=['hdbscan'],
                verbose=True
            )
            print(results)
        """
        result_dictlist: list[dict] = []
        np.random.seed(self.verbose+2)  # somehow seed this process

        for sample_size in sample_sizes:
            print('----------------------')
            print(f'Evaluating sample size {sample_size}!')

            # draw a sample of the appropriate size
            sample_indices: np.ndarray = np.random.choice(len(texts), size=sample_size, replace=False).tolist()
            sample: list[str] = [texts[i] for i in sample_indices]
            embeddings: np.ndarray = self.embeddings[sample_indices]
            reduced_embeddings: np.ndarray = self.reduced_embeddings[sample_indices]

            # and now do the same evaluation as before
            for clustering_algorithm in clustering_algorithms:
                print('.................')
                print(f'Utilising {clustering_algorithm} for clustering.')
                # and over all nr_topic parameters
                for nr_topics in params_nr_topics:
                    if clustering_algorithm == 'hdbscan' or nr_topics != 'auto':
                        print(f'Running topic model with nr_topics={nr_topics}')
                        topics, train_tc, train_td, _ = self.fit_transform(
                            texts=sample,
                            nr_topics=nr_topics,
                            stop_words=stop_words,
                            clustering_algorithm=clustering_algorithm,
                            min_cluster_size=min_cluster_size,
                            zeroshot_topic_list=zeroshot_topic_list,
                            zeroshot_min_similarity=zeroshot_min_similarity,
                            verbose=verbose,
                            embeddings=embeddings,
                            reduced_embeddings=reduced_embeddings
                        )

                        # evaluate topic quality on the test data set
                        topic_keywords: list[list[str]] = topics['Representation'].tolist()
                        test_tc: float = compute_avg_topic_npmi(documents=evaluation_texts,
                                                                topics=topic_keywords,
                                                                use_ppmi=True,
                                                                verbose=verbose)
                        test_td: float = compute_topic_diversity(topics=topic_keywords)

                        result_dictlist.append({
                            'clustering_algorithm': clustering_algorithm,
                            'nr_topics': nr_topics,
                            'actual_topics': topics.shape[0],
                            'train_size': len(sample),
                            'test_size': len(evaluation_texts),
                            'test_coherence': test_tc,
                            'test_diversity': test_td,
                            'test_quality': test_tc * test_td,
                            'train_coherence': train_tc,
                            'train_diversity': train_td,
                            'train_quality': train_tc * train_td
                        })
        return pd.DataFrame.from_dict(result_dictlist)

    def parallel_sampled_topic_evaluation(
            self, texts: list[str],
            evaluation_texts: list[str],
            params_nr_topics: list[int],
            sample_sizes: list[int],
            clustering_algorithms: list[str] = ['hdbscan', 'kmeans'],
            stop_words: list[str] = stopwords.words('english'),
            min_cluster_size: int = 100,
            zeroshot_topic_list: list[str] | None = None,
            zeroshot_min_similarity: float = 0.5,
            verbose: bool = True,
            max_workers: int | None = None) -> pd.DataFrame:
        """
        Perform parallel evaluation of sampled topic modelling.

        Args:
           texts (list[str]): The training dataset of text samples for topic modeling.
            evaluation_texts (list[str]): The test dataset used to evaluate topic quality.
            params_nr_topics (list[int]): A list of numbers representing the potential topic counts to evaluate.
            sample_sizes (list[int]): List of sample sizes to draw from the training texts.
            clustering_algorithms (list[str], optional): A list of clustering algorithms to test.
                Supported options are 'hdbscan' and 'kmeans'. Defaults to ['hdbscan', 'kmeans'].
            stop_words (list[str], optional): List of stop words to exclude during topic vectorization.
                Defaults to `stopwords.words('english')`.
            min_cluster_size (int, optional): Minimum cluster size for the HDBSCAN clustering algorithm.
                Ignored for k-means. Defaults to 100.
            zeroshot_topic_list (list[str] | None, optional): A predefined list of zero-shot topics for clustering guidance.
                If `None`, topics are inferred. Defaults to `None`.
            zeroshot_min_similarity (float, optional): Minimum cosine similarity threshold for assigning zero-shot topics.
                Defaults to 0.5.
            verbose (bool, optional): Whether to display progress and debug messages. Defaults to `True`.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the results of the evaluations.
        """
        results = []

        # Run evaluations within individual processes
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks to execute for each sample size and topic number
            futures_to_params_dict: dict = {}  # store the associations between futures and parameter configurations
            for clustering_algorithm in clustering_algorithms:
                for sample_size in sample_sizes:
                    for nr_topics in params_nr_topics:
                        futures_to_params_dict[executor.submit(self.evaluate_sampled_topic_model,
                                                               texts=texts,
                                                               evaluation_texts=evaluation_texts,
                                                               params_nr_topics=[nr_topics],
                                                               sample_sizes=[sample_size],
                                                               clustering_algorithms=[clustering_algorithm],
                                                               stop_words=stop_words,
                                                               min_cluster_size=min_cluster_size,
                                                               zeroshot_topic_list=zeroshot_topic_list,
                                                               zeroshot_min_similarity=zeroshot_min_similarity,
                                                               verbose=verbose)] = (sample_size, nr_topics, clustering_algorithm)

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures_to_params_dict):
                sample_size, nr_topics, clustering_algorithm = futures_to_params_dict[future]  # read parameters from task
                try:
                    result = future.result()  # Get the result
                    results.append(result)
                except Exception as e:
                    print(f"Configuration ({sample_size}, {nr_topics}, {clustering_algorithm}) failed with exception: {e}")

        # Combine all results into a single DataFrame (if applicable)
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no results

    @staticmethod
    def retrieve_zero_shot_samples(queries: list[str], texts: list[str], threshold: int = 0.7,
                                   n_samples: int = 10, verbose: bool = True,
                                   embedding_model: str = 'Alibaba-NLP/gte-multilingual-base') -> pd.DataFrame:
        """
        Retrieves text samples that are semantically similar to given queries using a zero-shot approach.

        Args:
            queries (list[str]): A list of query strings to compare against the provided texts.
            texts (list[str]): A list of text samples to evaluate for semantic similarity.
            threshold (int, optional): The minimum cosine similarity score for a text to be considered relevant.
                Defaults to 0.7.
            n_samples (int, optional): The maximum number of samples to return for each query. Defaults to 10.
            verbose (bool, optional): If True, displays a progress bar while encoding texts and queries.
                Defaults to True.
            embedding_model (str, optional): The name of the SentenceTransformer model used for generating embeddings.
                Defaults to 'Alibaba-NLP/gte-multilingual-base'.

        Returns:
            pd.DataFrame: A DataFrame where each row contains a query and its corresponding list of relevant text samples.
            The DataFrame has two columns:
                - 'query': The query string.
                - 'samples': A list of text samples with similarity above the specified threshold, limited to `n_samples`.
        """
        # extract embeddings for both queries and texts
        sentence_model: SentenceTransformer = SentenceTransformer(embedding_model, trust_remote_code=True)
        text_embeddings: np.ndarray = sentence_model.encode(texts, show_progress_bar=verbose)
        query_embeddings: np.ndarray = sentence_model.encode(queries, show_progress_bar=verbose)

        # compute the cosine similarity
        sim_matrix: np.ndarray = cosine_similarity(query_embeddings, text_embeddings)

        # for each query, get at least n_samples indices with similarity > threshold
        result_dictlist: list[dict] = []
        for i, query in enumerate(queries):
            queried_samples: np.ndarray = np.argwhere(sim_matrix[i] > threshold).flatten()
            result_dictlist.append({
                'query': query,
                'samples': [texts[j] for j in queried_samples[:n_samples]]
            })
        return pd.DataFrame.from_dict(result_dictlist)

    @staticmethod
    def evaluate_zero_shot_accuaracy(queries: list[str], texts: list[str], labels: list[int],
                                     threshold: int = 0.7, verbose: bool = True,
                                     embedding_model: str = 'Alibaba-NLP/gte-multilingual-base'
                                     ) -> Tuple[float, list[int]]:
        """
        Evaluates the zero-shot classification accuracy of texts against queries based on cosine similarity.

        Args:
            queries (list[str]): A list of query strings used as the classification reference points.
            texts (list[str]): A list of text samples to be classified.
            labels (list[int]): A list of true labels for the texts, where each label corresponds to the index
                of the correct query in the `queries` list. Labels must range from 0 to len(queries)-1.
            threshold (int, optional): The minimum cosine similarity score required for a prediction. If no
                query exceeds this threshold for a given text, the method predicts -1 for that text. Defaults to 0.7.
            verbose (bool, optional): If True, displays a progress bar while encoding texts and queries. Defaults to True.
            embedding_model (str, optional): The name of the SentenceTransformer model used for generating embeddings.
                Defaults to 'Alibaba-NLP/gte-multilingual-base'.

        Returns:
            Tuple[float, list[int]]:
                - float: The classification accuracy, calculated as the proportion of correctly predicted labels.
                - list[int]: A list of predicted labels for the texts. If a text does not match any query above the
                threshold, its prediction will be -1.
        """
        # note: labels must be 0-n where i corresponds to queries[i]

        # extract embeddings for both queries and texts
        sentence_model: SentenceTransformer = SentenceTransformer(embedding_model)
        text_embeddings: np.ndarray = sentence_model.encode(texts, show_progress_bar=verbose)
        query_embeddings: np.ndarray = sentence_model.encode(queries, show_progress_bar=verbose)

        # compute the cosine similarity
        sim_matrix: np.ndarray = cosine_similarity(text_embeddings, query_embeddings)

        # make a prediction for each input text
        predictions: list[int] = []
        for i, text in texts:
            similarities: np.ndarray = sim_matrix[i]
            candidate_idx: int = np.argmax(similarities)

            # check if similarity exceeds the threshold
            if similarities[candidate_idx] > threshold:
                predictions.append(candidate_idx)
            # if threshold is not exceeded, append -1
            else:
                predictions.append(-1)

        # evaluate the classification accuracy
        accucary: float = accuracy_score(y_true=labels, y_pred=predictions)

        # return accuracy and predictions
        return accucary, predictions


class PrecomputedUMAP:
    """Custom class that returns the precomputed reduced
    vector for each high-dim. embedding vector.
    """
    def __init__(self, embeddings: np.ndarray, reduced_embeddings: np.ndarray):
        """Builds a dictionary where the raw embedding bytes act as keys and
        the reduced embeddings are the values.
        """
        self.precomputed_embeddings: dict = {}
        for i in range(embeddings.shape[0]):
            self.precomputed_embeddings[embeddings[i].tobytes()] = reduced_embeddings[i]

    def fit(self, X: np.ndarray):
        """Empty fit function as it is already fitted."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Retrieve the reduced embedding given input embeddings.
        """
        retrieved_embeddings: list[np.ndarray] = []
        for i in range(X.shape[0]):
            retrieved_embeddings.append(self.precomputed_embeddings[X[i].tobytes()])
        return np.stack(retrieved_embeddings, axis=0)
