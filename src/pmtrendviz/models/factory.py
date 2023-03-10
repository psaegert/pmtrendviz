import inspect
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Tuple

from ..models import manager as mgr
from ..utils import download_model, get_models_dir
from .manager import ModelManager

logger = logging.getLogger('pmtrendviz.models.factory')
CPU_COUNT = os.cpu_count() or 1


class ModelFactory(object):
    @staticmethod
    def create(
        model: str,
        include_title: bool = True,
        include_abstract: bool = True,
        include_keywords_major: bool = False,
        include_keywords_minor: bool = False,
        **kwargs: Any
    ) -> ModelManager:
        """
        Create a model manager from the given arguments

        Parameters
        ----------
        model : str
            The name of the model to create. This must correspond to a method in the ModelFactory class.
        include_title : bool
            Whether to include the title in the input text
        include_abstract : bool
            Whether to include the abstract in the input text
        include_keywords_major : bool
            Whether to include the major keywords in the input text
        include_keywords_minor : bool
            Whether to include the minor keywords in the input text
        **kwargs : Any
            The keyword arguments to pass to the ModelFactory method

        Returns
        -------
        ModelManager
            The corresponding model manager
        """
        # Check if the ModelFactory has a method with the given name
        if not hasattr(ModelFactory, model):
            logger.error(f'No model named {model} found in ModelFactory.')
            raise ValueError(f'No model named {model} found in ModelFactory.')

        # Get the method
        method = getattr(ModelFactory, model)

        # Collect the preprocessing kwargs
        preprocessing_kwargs = {
            'include_title': include_title,
            'include_abstract': include_abstract,
            'include_keywords_major': include_keywords_major,
            'include_keywords_minor': include_keywords_minor
        }

        logger.debug(f'Creating model {model} with preprocessing kwargs {preprocessing_kwargs} and kwargs {kwargs}')

        # Filter out the kwargs that are not needed for the model with the inspect module
        model_kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(method).parameters}

        # Call the method
        return method(preprocessing_kwargs, **model_kwargs)

    @staticmethod
    def load(name: str) -> ModelManager:
        """
        Load a model from the models directory

        Parameters
        ----------
        name : str
            The name of the model to load

        Returns
        -------
        ModelManager
            The model manager
        """
        load_from_dir = os.path.join(get_models_dir(), name)

        # Read the config file
        with open(os.path.join(load_from_dir, 'config.json'), 'r') as f:
            config = json.load(f)

        # Call the load method of the corresponding manager
        manager_class_name = config['manager']

        logger.debug(f'Loading model {name} with manager {manager_class_name}')

        # Get the class
        manager_class = getattr(mgr, manager_class_name)

        # Call the load method
        return manager_class.load(name)

    @staticmethod
    def remove(name: str, ignore_errors: bool = False) -> None:
        """
        Remove a model from the models directory

        Parameters
        ----------
        name : str
            The name of the model to remove
        ignore_errors : bool, optional
            Whether to ignore errors when removing the model, by default False
        """
        from ..utils import get_es_client

        model_dir = os.path.join(get_models_dir(), name)

        logger.info(f'Removing model {name}.')

        # Remove the directory
        shutil.rmtree(model_dir, ignore_errors=ignore_errors)

        # Remove the predictions indices
        es = get_es_client()

        # Get the indices of the predictions of the model for all source indices (e.g. 'pubmed_test-model_predictions')
        indices = es.indices.get_alias(index=f'*_{name}_predictions')

        logger.info(f'Deleting {len(indices)} indices for model {name}.')

        # Delete the indices
        for index in indices:
            logger.debug(f'Deleting index {index}.')
            es.indices.delete(index=index, ignore_unavailable=ignore_errors)

    @staticmethod
    def install(name: str, overwrite: bool = False) -> None:
        """
        Install a pmtrendviz model from the Huggingface Hub

        Parameters
        ----------
        name : str
            The name of the model to install
        overwrite : bool, optional
            Whether to overwrite the model if it already exists, by default False
        """
        download_model(
            author='psaegert',
            model=f'pmtrendviz-{name}',
            target_dir=os.path.join(get_models_dir(), name),
            overwrite=overwrite)

        # Load the model to import the predictions into Elasticsearch
        logger.debug(f'Loading model {name} to import predictions.')
        manager = ModelFactory.load(name)

        # For every file in the model's directory ending with '_predictions.json', import the predictions into Elasticsearch
        for file in os.listdir(os.path.join(get_models_dir(), name)):
            if file.endswith('_predictions.json'):
                logger.debug(f'Importing predictions from file {file}.')

                manager.import_predictions(filepath=os.path.join(get_models_dir(), name, file))

                # Delete the file after importing the predictions
                logger.debug(f'Deleting file {file}.')
                os.remove(os.path.join(get_models_dir(), name, file))

    @staticmethod
    def tfidf_truncatedsvd_kmeans(
            preprocessing_kwargs: Dict[str, bool] | None = None,
            stop_words: str | None = None,
            max_df: float = 1.0,
            min_df: float = 1,
            ngram_range: Tuple[int, int] = (1, 1),
            n_components: int = 100,
            n_clusters: int = 100,
            random_state: int | None = None,
    ) -> ModelManager:
        """
        Create a SimpleSklearnPipelineManager with a TfidfVectorizer, TruncatedSVD and MiniBatchKMeans

        Parameters
        ----------
        preprocessing_kwargs : Dict[str, bool], optional
            Keyword arguments for the preprocessing, by default None
        stop_words : str, optional
            The stop words to use for the TfidfVectorizer, by default None
        max_df : float, optional
            The maximum document frequency of the TfidfVectorizer, by default 1.0
        min_df : float, optional
            The minimum document frequency of the TfidfVectorizer, by default 1
        ngram_range : Tuple[int, int], optional
            The ngram range of the TfidfVectorizer, by default (1, 1)
        n_components : int, optional
            The number of components of the TruncatedSVD, by default 100
        n_clusters : int, optional
            The number of clusters of the TruncatedSVD, by default 100
        random_state : int, optional
            The random state of the TruncatedSVD and MiniBatchKMeans, by default None

        Returns
        -------
        SimpleSklearnPipelineManager
            The model manager
        """
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline

        from .manager import SimpleSklearnPipelineManager

        # HACK: The manager needs the prefixes in the transform method
        pipeline = Pipeline([
            ('vectorize_tfidf', TfidfVectorizer(stop_words=stop_words, max_df=max_df, min_df=min_df, ngram_range=tuple(ngram_range))),
            ('transform_svd', TruncatedSVD(n_components=n_components, random_state=random_state)),
            ('clustering_kmeans', MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=256 * CPU_COUNT, n_init='auto')),
        ], verbose=True)

        return SimpleSklearnPipelineManager(pipeline, preprocessing_kwargs, random_state)

    @staticmethod
    def ner_tfidf_truncatedsvd_kmeans(
            preprocessing_kwargs: Dict[str, bool] | None = None,
            stop_words: str | None = None,
            max_df: float = 1.0,
            min_df: float = 1,
            ngram_range: Tuple[int, int] = (1, 1),
            n_components: int = 100,
            n_clusters: int = 100,
            spacy_model: str = 'en_core_web_sm',
            spacy_disable: List[str] | None = None,
            random_state: int | None = None,
    ) -> ModelManager:
        """
        Create a SpacySklearnPipelineManager with a TfidfVectorizer, TruncatedSVD and MiniBatchKMeans

        Parameters
        ----------
        preprocessing_kwargs : Dict[str, bool], optional
            Keyword arguments for the preprocessing, by default None
        stop_words : str, optional
            The stop words to use for the TfidfVectorizer, by default None
        max_df : float, optional
            The maximum document frequency of the TfidfVectorizer, by default 1.0
        min_df : float, optional
            The minimum document frequency of the TfidfVectorizer, by default 1
        ngram_range : Tuple[int, int], optional
            The ngram range of the TfidfVectorizer, by default (1, 1)
        n_components : int, optional
            The number of components of the TruncatedSVD, by default 100
        n_clusters : int, optional
            The number of clusters of the TruncatedSVD, by default 100
        spacy_model : str, optional
            The spacy model to use for Named Entity Recognition, by default 'en_core_web_sm'
        spacy_disable : List[str], optional
            The spacy pipeline components to disable, by default None

        Returns
        -------
        SpacySklearnPipelineManager
            The model manager
        """
        import spacy
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline

        from .manager import SpacySklearnPipelineManager

        # HACK: The manager needs the prefixes in the transform method
        pipeline = Pipeline([
            ('vectorize_tfidf', TfidfVectorizer(stop_words=stop_words, max_df=max_df, min_df=min_df, ngram_range=tuple(ngram_range))),
            ('transform_svd', TruncatedSVD(n_components=n_components, random_state=random_state)),
            ('clustering_kmeans', MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=256 * CPU_COUNT, n_init='auto')),
        ], verbose=True)

        if spacy_disable is not None:
            nlp = spacy.load(str(spacy_model), disable=spacy_disable)
        else:
            nlp = spacy.load(spacy_model)

        return SpacySklearnPipelineManager(pipeline, nlp, preprocessing_kwargs, random_state)

    @staticmethod
    def default_bertopic(
            preprocessing_kwargs: Dict[str, bool] | None = None,
    ) -> ModelManager:
        """
        Create a BertopicManager

        Parameters
        ----------
        preprocessing_kwargs : Dict[str, bool], optional
            Keyword arguments for the preprocessing, by default None

        Returns
        -------
        BertopicManager
            The model manager
        """
        from bertopic import BERTopic

        from .manager import BERTopicManager

        bertopic = BERTopic(verbose=True)

        return BERTopicManager(bertopic, preprocessing_kwargs)

    @staticmethod
    def default_top2vec(
            preprocessing_kwargs: Dict[str, bool] | None = None,
    ) -> ModelManager:
        """
        Create a Top2VecManager

        Parameters
        ----------
        preprocessing_kwargs : Dict[str, bool], optional
            Keyword arguments for the preprocessing, by default None
<
        Returns
        -------
        Top2VecManager
            The model manager
        """
        from .manager import Top2VecManager

        top2vec = None

        return Top2VecManager(top2vec, preprocessing_kwargs)
