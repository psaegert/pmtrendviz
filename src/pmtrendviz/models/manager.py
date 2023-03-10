import json
import logging
import os
import time
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd
import spacy
from bertopic import BERTopic
from pandas import Series
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from top2vec import Top2Vec
from top2vec.Top2Vec import default_tokenizer
from tqdm import tqdm

from elasticsearch import Elasticsearch, helpers

from ..import_pipeline.index import import_file
from ..train.preprocessing import combine_text
from ..utils import check_name, construct_predictions_file_name, construct_predictions_index_name, \
    get_articles_per_date, get_es_client, get_models_dir, get_time_to_complete_offset, \
    reconstruct_document_date

logger = logging.getLogger('pmtrendviz.models.manager')

# Set the log level of the SentenceTransformer module to the pmtrendviz log level
if logging.getLogger('pmtrendviz').isEnabledFor(logging.DEBUG):
    logging.getLogger('SentenceTransformer').setLevel(logging.DEBUG)
else:
    # Disable the 'Batches' progress bar of the SentenceTransformer module
    logging.getLogger('SentenceTransformer').setLevel(logging.WARNING)


class ModelManager():
    def __init__(self, model: Any, preprocessing_args: Dict[str, bool] | None = None, random_state: int | None = None) -> None:
        """
        Initialize the model manager

        Parameters
        ----------
        model : Any
            The model to use
        preprocessing_args : Dict[str, bool]
            The preprocessing arguments to use
        random_state : int | None, optional
            The random state to use, by default None
        """
        self.model = model
        self._name: str | None = None  # Set by the child class during 'load' or 'save'. Used to allow emergency saving from within the model's methods
        self.random_state = random_state
        self.preprocessing_args = preprocessing_args
        self.new_predictions_buffer_dict: Dict[int, Dict[str, Union[str | None, int]]] = {}
        self.cluster_names = pd.DataFrame(columns=['name'])

    @abstractmethod
    def fit(self, X: Union[List[str], pd.Series]) -> None:
        """
        Fit the model

        Parameters
        ----------
        X : Union[List[str], pd.Series]
            The documents to fit the model on
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Predict the clusters for the documents

        Parameters
        ----------
        X : Union[List[str], pd.Series]
            The documents to predict the clusters for
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: Union[List[str], pd.Series], last_step: str | None = None, exclude_from: str | None = None) -> np.ndarray:
        """
        Transform the documents into a vector space

        Parameters
        ----------
        X : Union[List[str], pd.Series]
            The documents to transform
        last_step : str | None, optional
            The last step to perform, by default None (all steps)
        exclude_from : str | None, optional
            The step to exclude from, by default None (no steps)
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, name: str, overwrite: bool = False, save_only: Union[str, List[str]] | None = None) -> None:
        """
        Save the model to the models directory

        Parameters
        ----------
        name : str
            The name of the model
        overwrite : bool, optional
            Whether to overwrite the model if it already exists, by default False
        save_only : List[str] | None, optional
            The only parts to save, by default None (save everything)
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(name: str) -> Any:
        """
        Load a model from the models directory

        Parameters
        ----------
        name : str
            The name of the model

        Returns
        -------
        ModelManager
            The model manager
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, X: Union[List[str], pd.Series], y: np.ndarray, scores: Dict[str, Callable[[Any, Any], float]] | None = None) -> pd.DataFrame:
        """
        Evaluate the model by calculating the given scores

        Parameters
        ----------
        X : Union[List[str], pd.Series]
            The documents
        y : np.ndarray
            The labels
        scores : Dict[str, Callable[[Any, Any], float]] | None, optional
            The scores to compute, by default Silhouette, Calinski-Harabasz and Davies-Bouldin

        Returns
        -------
        pd.DataFrame
            The scores
        """
        raise NotImplementedError

    def set_name(self, name: str) -> None:
        """
        Set the name of the model. Only possible if the model has not been saved yet.

        Parameters
        ----------
        name: str
            The name of the model
        """
        logger.debug(f'Setting the name of the model from {self._name} to {name}')

        if name is None:
            logger.error('The name of the model cannot be None')
            raise ValueError('The name of the model cannot be None')

        if self._name == name:
            return

        if self._name is not None:
            logger.error('The name of the model has already been set and cannot be changed, as it would require reindexing the predictions. Please consider omitting the name argument.')
            raise ValueError('The name of the model has already been set and cannot be changed, as it would require reindexing the predictions. Please consider omitting the name argument.')

        check_name(name)

        self._name = name

    def precompute_predictions(self, source_index: str, max_new_predictions: int | None = None, timeout: int | None = None, batch_size: int = 1, method: str = 'uniform', overwrite: bool = False, es: Elasticsearch | None = None) -> None:
        """
        Precompute the predictions for the model

        Paremeters
        ----------
        source_index : str
            The preprocessor to use
        max_new_predictions : int | None, optional
            The maximum number of new predictions to make, by default None (unlimited)
        timeout : int | None, optional
            The timeout in seconds, by default None (unlimited)
        batch_size : int, optional
            The batch size to use, by default 1
        method : {'uniform', 'forward'}, optional
            The order in which to precompute the predictions, by default 'uniform'
            'uniform' - Randomly select documents from the index. Slow but can be stopped early without concerns.
            'forward' - Select documents in the order they are stored in the index. WARNING: This is significantly faster but has a drastic effect on the trend visualization and should only be considered when precomputing the predictions of the entire index.
        overwrite : bool, optional
            Whether to overwrite the existing predictions, by default False
        es : Elasticsearch | None, optional
            The Elasticsearch client, by default, a new client will be created
        """
        # Check the arguments
        if timeout is not None and timeout <= 0:
            logger.error(f'The timeout must be greater than 0, got {timeout} instead')
            raise ValueError(f'The timeout must be greater than 0, got {timeout} instead')

        if max_new_predictions is not None and max_new_predictions <= 0:
            logger.error(f'The maximum number of new predictions must be greater than 0, got {max_new_predictions} instead')
            raise ValueError(f'The maximum number of new predictions must be greater than 0, got {max_new_predictions} instead')

        if es is None:
            es = get_es_client()

        # Define the query for sampling new articles to predict
        # Limit the scroll time to 3 hours for one batch
        scroll_time = np.clip(batch_size // 10, 1, 60 * 3)
        match method:
            case 'uniform':
                # Randomly select documents from the source_index (slow but can be stopped early without concerns)
                scanner = helpers.scan(es, index=source_index, query={'query': {'function_score': {'random_score': {}}}}, scroll=f'{scroll_time}m', preserve_order=True)
            case 'forward':
                # Select documents in the order they are stored in the source_index (fast but must be done until completion to avoid bias)
                scanner = helpers.scan(es, index=source_index, query={'query': {'match_all': {}}}, scroll=f'{scroll_time}m')
            case _:
                logger.error(f'Invalid method: {method}')
                raise ValueError(f'Invalid method: {method}')

        # Construct the name of the predictions index
        predictions_index_name = self._construct_predictions_index_name(source_index)

        # Check if the predictions index exists
        if not es.indices.exists(index=predictions_index_name):
            self.create_predictions_index(predictions_index_name, es)

        # Count the number of documents in the source index
        source_count = int(es.cat.count(index=source_index, params={"format": "json"})[0]['count'])  # type: ignore

        # Count the number of documents in the predictions index
        predictions_count = int(es.cat.count(index=predictions_index_name, params={"format": "json"})[0]['count'])  # type: ignore

        # Get all the PMIDs of the already predicted articles
        # Do not get the source, just the ids
        query = {"query": {"match_all": {}}, "_source": False}

        # Get the ids
        already_predicted_pmids = np.empty(predictions_count, dtype=np.int64)
        for i, hit in enumerate(tqdm(helpers.scan(es, index=predictions_index_name, query=query), total=predictions_count, desc='Filtering done predictions')):
            already_predicted_pmids[i] = int(hit['_id'])

        # Convert the already predicted PMIDs to a set for faster lookup
        already_predicted_pmids = set(already_predicted_pmids)  # type: ignore

        # Get the time_to_complete_offset to reconstruct and unify dates
        time_to_complete_offset = get_time_to_complete_offset(source_index, es)

        # Log the start of the precomputation. This needs to be done before the progress bar is created, otherwise the progressbar will be split into two lines
        print(f'Precomputing predictions in {method} mode...')

        # Create the progress bar if logging is enabled for WARNING. Widen the smoothing to give more accurate time estimates
        pbar = tqdm(scanner, total=source_count, disable=not logger.isEnabledFor(logging.WARNING), smoothing=0.05)

        # Create a dataframe of length 'count - len(already_predicted_articles_PMIDS)' to store the new predictions
        batch_predictions = pd.DataFrame(index=range(batch_size), columns=['PMID', 'date', 'date_completed', 'label', 'text'])

        # Iterate over all the documents in the source_index # HACK: This still appears to be the most efficient way
        if timeout is not None:
            start_time = time.time()

        # Keep track of the number of new predictions and use it as the index for the batch_predictions dataframe
        new_prediction_count = 0

        # Keep track of the indices of the documents that need to be predicted
        # This improves efficiency
        indices_to_predict = []

        # Iterate over all the documents in the source index
        for doc in pbar:

            # Check if the document already exists in the predictions index
            if int(doc['_id']) in already_predicted_pmids and not overwrite:
                continue

            # Combine the text
            text = combine_text(doc, **self.preprocessing_args if self.preprocessing_args is not None else {})

            # Reconstruct the date of the article, since some articles have missing dates
            date = reconstruct_document_date(doc, source_index, time_to_complete_offset)

            # Add the document to the list of new predictions
            # The label will be added later
            batch_predictions.loc[new_prediction_count % batch_size, 'PMID'] = int(doc['_id'])
            batch_predictions.loc[new_prediction_count % batch_size, 'date'] = date
            batch_predictions.loc[new_prediction_count % batch_size, 'date_completed'] = doc['_source']['DATE_COMPLETED']
            batch_predictions.loc[new_prediction_count % batch_size, 'text'] = text

            # Add the index to the list of indices to predict
            indices_to_predict.append(new_prediction_count % batch_size)

            # Increment the number of new predictions
            new_prediction_count += 1

            # Check if the batch size has been reached
            batch_size_reached = len(indices_to_predict) >= batch_size
            max_new_predictions_reached = max_new_predictions is not None and new_prediction_count >= max_new_predictions
            timeout_reached = timeout is not None and time.time() - start_time >= timeout

            if batch_size_reached or max_new_predictions_reached or timeout_reached:

                # If the batch size is sufficiently large, indicate that the batch is being processed
                if batch_size > 10: pbar.set_description('predicting batch')  # NOQA: E701

                # Predict the labels and add them to the batch_predictions dataframe
                batch_predictions = self._precompute_new_predictions(batch_predictions, indices_to_predict)

                # Add the new predictions to the predictions index
                self._add_predictions_to_index(batch_predictions, predictions_index_name, es)

                # Reset the list of indices to predict
                indices_to_predict = []

            # If the batch size is sufficiently large, indicate that new data is being fetched
            if batch_size > 10: pbar.set_description('fetching data')  # NOQA: E701

            # Check the stopping conditions
            if max_new_predictions_reached or timeout_reached:
                break

        # If there are any remaining predictions, predict them
        if len(indices_to_predict) > 0:
            # If the batch size is sufficiently large, indicate that the batch is being processed
            if batch_size > 10: pbar.set_description('predicting batch')  # NOQA: E701

            # Predict the labels and add them to the batch_predictions dataframe
            batch_predictions = self._precompute_new_predictions(batch_predictions, indices_to_predict)

            # Add the new predictions to the predictions index
            self._add_predictions_to_index(batch_predictions, predictions_index_name, es)

    def create_predictions_index(self, predictions_index_name: str, es: Elasticsearch = None) -> None:
        """
        Create an index to store the predictions of the model

        Parameters
        ----------
        predictions_index_name: str
            The name of the predictions index
        es: Elasticsearch, optional
            The Elasticsearch instance to use. If None, a new instance will be created
        """
        logger.info(f'Creating the predictions index {predictions_index_name}')

        if es is None:
            es = get_es_client()

        # Create the predictions index
        es.indices.create(index=predictions_index_name, mappings={
            'properties': {
                'PMID': {'type': 'keyword'},
                'date': {'type': 'date'},
                'date_completed': {'type': 'date'},
                'label': {'type': 'integer'}
            }}, settings={
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                }
        })

    @staticmethod
    def delete_predictions_index(predictions_index: str, es: Elasticsearch | None = None) -> None:
        """
        Delete the predictions index. Can be called without a ModelManager instance

        Parameters
        ----------
        predictions_index: str
            The name of the predictions index
        es: Elasticsearch | None, optional
            The Elasticsearch instance to use. If None, a new instance will be created
        """
        logger.info(f'Deleting the predictions index {predictions_index}')

        if es is None:
            es = get_es_client()

        # Delete the predictions index
        es.indices.delete(index=predictions_index, ignore=[400, 404])

    def clear_predictions(self, source_index: str, es: Elasticsearch | None = None) -> None:
        """
        Clear the predictions index of the model

        Parameters
        ----------
        source_index: str
            The name of the source index the predictions are based on
        es: Elasticsearch | None, optional
            The Elasticsearch instance to use. If None, a new instance will be created
        """
        logger.info(f'Clearing the predictions for {source_index}')

        if es is None:
            es = get_es_client()

        predictions_index_name = self._construct_predictions_index_name(source_index)

        # Delete the predictions index using the static method
        self.delete_predictions_index(predictions_index_name, es)

    def export_predictions(self, source_index: str, es: Elasticsearch | None = None) -> None:
        """
        Export the predictions to a JSON file that can be used to bulk import the predictions into another Elasticsearch instance

        Parameters
        ----------
        source_index: str
            The name of the source index the predictions are based on
        es: Elasticsearch | None, optional
            The Elasticsearch instance to use. If None, a new instance will be created
        """
        logger.info(f'Exporting the predictions for source index {source_index}')

        if es is None:
            es = get_es_client()

        if self._name is None:
            logger.error('Cannot export predictions without a name. Call set_name() first')
            raise ValueError('Cannot export predictions without a name. Call set_name() first')

        predictions_index_name = self._construct_predictions_index_name(source_index)
        predictions_file_name = self._construct_predictions_file_name(source_index)

        # Check if the predictions index exists
        if not es.indices.exists(index=predictions_index_name):
            logger.error(f'Predictions index {predictions_index_name} does not exist. Precompute the predictions first.')
            raise ValueError(f'Predictions index {predictions_index_name} does not exist. Precompute the predictions first.')

        filepath = os.path.join(get_models_dir(), self._name, predictions_file_name)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Create a list of dictionaries containing the predictions
        logger.info(f'Exporting the predictions from predictions index {predictions_index_name} to file {filepath}')
        with open(filepath, 'w') as f:
            for hit in helpers.scan(es, index=predictions_index_name, query={'query': {'match_all': {}}}):
                f.write(json.dumps(hit['_source']) + '\n')

    def import_predictions(self, source_index: str | None = None, filepath: str | None = None, es: Elasticsearch | None = None) -> None:
        """
        Import the predictions from an external JSON file (e.g. downloaded from the pretrained models repository)

        Parameters
        ----------
        source_index: str | None, optional
            The name of the source index the predictions are based on, if None, the name will be extracted from the filepath
        filepath: str | None, optional
            The name of the file to import the predictions from. If None, the default filepath will be used
        es: Elasticsearch | None, optional
            The Elasticsearch instance to use. If None, a new instance will be created
        """
        logger.info(f'Importing the predictions for source index {source_index}')

        if es is None:
            es = get_es_client()

        if self._name is None:
            logger.error('Cannot export predictions without a name. Call set_name() first or load the model from disk')
            raise ValueError('Cannot export predictions without a name. Call set_name() first or load the model from disk')

        # Make sure that either source_index or filepath is specified and that the other is not
        if source_index is None and filepath is None:
            logger.error('Either source_index or filepath must be specified')
            raise ValueError('Either source_index or filepath must be specified')

        if source_index is not None and filepath is not None:
            logger.error('Either source_index or filepath must be specified, not both')
            raise ValueError('Either source_index or filepath must be specified, not both')

        if filepath is None:
            # Construct the predictions index name and file name from the source index
            predictions_index_name = self._construct_predictions_index_name(source_index)  # type: ignore
            filepath = os.path.join(get_models_dir(), self._name, self._construct_predictions_file_name(source_index))  # type: ignore
            logger.debug(f'Constructed the predictions index name and file name from the source index {source_index}: {predictions_index_name} and {filepath}')
        else:
            # Construct the predictions index name and file name from the filepath
            predictions_file_name = os.path.basename(filepath)

            # Strip the base name of the '_predictions.json' suffix to get the source index name
            source_index = predictions_file_name[:-len('_predictions.json')]

            # Construct the predictions index name
            predictions_index_name = self._construct_predictions_index_name(source_index)

            logger.debug(f'Constructed the predictions index name and file name from the filepath {filepath}: {predictions_index_name} and {source_index}')

        # If the predictions index already exists, delete it
        if es.indices.exists(index=predictions_index_name):
            logger.debug(f'Predictions index {predictions_index_name} already exists. Deleting it first')
            self.delete_predictions_index(predictions_index_name, es)

        # Create the predictions index
        self.create_predictions_index(predictions_index_name, es)

        # Import the predictions from the file
        logger.info(f'Importing the predictions from file {filepath} to predictions index {predictions_index_name}')
        import_file(filepath, es, index=predictions_index_name, skip_odd=False)

    @abstractmethod
    def generate_trends(self, query: str, resolution: str, source_index: str, distance: str = 'cosine', closest_n: int = 1, ignore_top_n_dates: int = 0, es: Elasticsearch | None = None) -> pd.DataFrame:
        """
        Generate trends for the given query

        Parameters
        ----------
        query : str
            The query to use
        resolution : str
            The resolution of the trends
        source_index : str
            The index containing the abstracts
        distance : str, optional
            The distance to use, by default 'cosine'
        closest_n : int, optional
            The number of closest clusters to use, by default 1
        ignore_top_n_dates : int, optional
            The number of biggest spikes to ignore, by default 0
        es : Elasticsearch | None, optional
            The Elasticsearch client, by default None

        Returns
        -------
        pd.DataFrame
            The trends
        """
        raise NotImplementedError

    def _precompute_new_predictions(self, new_predictions: pd.DataFrame, indices_to_predict: List[int]) -> pd.DataFrame:
        """
        Precompute the labels for a batch of new predictions which is dictated by the given indices for the new predictions dataframe

        Parameters
        ----------
        new_predictions: pd.DataFrame
            The DataFrame containing the new predictions
        indices_to_predict: List[int]
            The indices of the new predictions to predict

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the new predictions with the labels added
        """
        # Check if there are any articles to predict
        if len(indices_to_predict) == 0:
            logger.warning('Called _precompute_new_predictions with no articles to predict')
            # Return the predictions DataFrame
            return new_predictions

        # Get the batch of articles to predict
        # Convert the Series to a list to avoid KeyErrors or IndexErrors
        new_predictions.loc[indices_to_predict, 'label'] = self.predict(list(new_predictions.loc[indices_to_predict, 'text']))

        # Remove the text from the rows that have been predicted to save memory
        new_predictions.loc[indices_to_predict, 'text'] = None

        return new_predictions

    def _add_predictions_to_index(self, new_predictions: pd.DataFrame, index: str, es: Elasticsearch) -> None:
        """
        Add the new predictions to the predictions index

        Parameters
        ----------
        new_predictions: pd.DataFrame
            The DataFrame containing the new predictions
        index: str
            The name of the index to add the predictions to
        es: Elasticsearch
            The Elasticsearch client
        """
        # Convert the new_predictions DataFrame to a list of dictionaries
        # The columns of the new_predicitons DataFrame are 'PMID', 'date', 'date_completed', 'label', 'text'
        # Only the 'PMID', 'date', 'date_completed', and 'label' columns are needed
        new_predictions = new_predictions[['PMID', 'date', 'date_completed', 'label']].to_dict(orient='records')

        logger.debug(f'Adding {len(new_predictions)} new predictions to the predictions index {index}: First 5: {new_predictions[:5]}')

        # Add the new predictions to the predictions index
        helpers.bulk(es, [{'_index': index, '_id': int(doc['PMID']), '_source': doc} for doc in new_predictions])

    def _compute_cluster_names(self, X: Union[List[str], pd.Series]) -> None:
        """
        Compute the cluster names

        Parameters
        ----------
        X : Union[List[str], Series]
            The documents
        y : np.ndarray
            The labels

        Returns
        -------
        Dict[int, str]
            The cluster names
        """
        raise NotImplementedError

    @abstractmethod
    def _preprocess(self, X: Union[List[str], pd.Series]) -> Union[List[str], pd.Series]:
        """
        Preprocess a batch of documents

        Parameters
        ----------
        X : Union[List[str], Series]
            The documents

        Returns
        -------
        Union[List[str], Series]
            The preprocessed documents
        """
        raise NotImplementedError

    @abstractmethod
    def _preprocess_single(self, x: str) -> str:
        """
        Preprocess a single document

        Parameters
        ----------
        x : str
            The document

        Returns
        -------
        str
            The preprocessed document
        """
        raise NotImplementedError

    def _compute_scores(self, X: Union[List[str], pd.Series], y: np.ndarray, scores: Dict[str, Callable[[Any, Any], float]] | None = None) -> pd.DataFrame:
        """
        Compute the scores

        Parameters
        ----------
        X : Union[List[str], pd.Series]
            The documents
        y : np.ndarray
            The labels
        scores : Dict[str, Callable[[Any, Any], float]] | None, optional
            The scores to compute, by default Silhouette, Calinski-Harabasz and Davies-Bouldin

        Returns
        -------
        pd.DataFrame
            The scores
        """
        if scores is None:
            scores = {
                'Silhouette': silhouette_score,
                'Calinski-Harabasz': calinski_harabasz_score,
                'Davies-Bouldin': davies_bouldin_score
            }

        score_values = {}

        for score_name, score_func in scores.items():
            logger.debug(f'Calculating the {score_name} score on {len(X)} documents')
            score_values[score_name] = score_func(X, y)

        return pd.DataFrame(score_values, index=[0])

    def _get_label_trend(self, source_index: str, label: int, resolution: str, ignore_top_n_dates: int = 0, es: Elasticsearch | None = None) -> pd.DataFrame:
        """
        Get the trend for the given label

        Parameters
        ----------
        source_index : str
            The name of the main index containing the articles
        label : int
            The label to get the trend for
        resolution : str
            The resolution of the trend
        ignore_top_n_dates : int, optional
            The number of biggest spikes to ignore, by default 0
        es : Elasticsearch | None, optional
            The Elasticsearch client, by default None

        Returns
        -------
        pd.DataFrame
            The trend for the given label
        """
        if es is None:
            es = get_es_client()

        # Get the distribution of articles per date
        articles_per_date = get_articles_per_date(index=source_index, es=es)

        if len(articles_per_date) == 0:
            # Do not filter out any spikes
            logger.warning(f'The source index {source_index} contains insufficient information to filter out the top n dates')
            query = {'bool': {'filter': {'term': {'label': label}}}}
        else:
            # Get the top n dates to ignore
            top_n_dates = articles_per_date.sort_values('rank', ascending=True).head(ignore_top_n_dates)['date'].tolist()
            top_n_dates_str = [date.strftime('%Y-%m-%d') for date in top_n_dates]

            query = {'bool': {'filter': {'term': {'label': label}}, 'must_not': [{'terms': {'date_completed': top_n_dates_str}}]}}  # type: ignore

        # Construct the name of the predictions index
        predictions_index = self._construct_predictions_index_name(source_index)

        # Query the predictions index for the given label and aggregate the results with a date_histogram aggregation
        aggs = {'trend': {'date_histogram': {'field': 'date', 'calendar_interval': resolution}}}

        logger.debug(f'Querying the predictions index {predictions_index} for label {label} with aggs {aggs} and query {query}')

        # Get the trend for the given label
        trend = es.search(index=predictions_index, query=query, aggs=aggs)['aggregations']['trend']['buckets']

        if len(trend) == 0:
            logger.warning(f'No trend found for label {label}')
            return pd.DataFrame(columns=['date', f'count_{label}'])

        # Convert the trend to a DataFrame
        trend = pd.DataFrame(trend)

        # Rename the columns
        trend.rename(columns={'key_as_string': 'date', 'doc_count': f'count_{label}'}, inplace=True)

        # Drop the key column
        trend.drop(columns='key', inplace=True)

        # Convert the date column to a datetime object and ignore the timezones
        trend['date'] = pd.to_datetime(trend['date']).dt.tz_localize(None)

        return trend

    def _construct_predictions_file_name(self, source_index: str) -> str:
        """
        Construct the name of the predictions file for the given source index

        Parameters
        ----------
        source_index: str
            The name of the source index

        Returns
        -------
        str
            The name of the predictions file
        """
        return construct_predictions_file_name(source_index)

    def _construct_predictions_index_name(self, source_index: str) -> str:
        """
        Construct the name of the predictions index for the given source index

        Parameters
        ----------
        source_index: str
            The name of the source index

        Returns
        -------
        str
            The name of the predictions index
        """
        if self._name is None:
            logger.error('Cannot construct predictions index name without a name. Call set_name() first')
            raise ValueError('Cannot construct predictions index name without a name. Call set_name() first')

        return construct_predictions_index_name(source_index, self._name)


class SimpleSklearnPipelineManager(ModelManager):
    def __init__(self, model: Pipeline, preprocessing_args: Dict[str, bool] | None = None, random_state: int | None = None) -> None:
        """
        Initialize the model manager for TF-IDF-based pipelines

        Parameters
        ----------
        model : Pipeline
            The model to use
        preprocessing_args : Dict[str, bool] | None, optional
            The preprocessing arguments to use
        random_state : int | None, optional
            The random state to use, by default None
        """
        super().__init__(model, preprocessing_args, random_state)
        self.model = model

    def fit(self, X: Union[List[str], pd.Series]) -> None:
        """
        Fit the model

        Parameters
        ----------
        X : Union[List[str], pd.Series]
            The documents
        """
        logger.debug('Fitting the model')
        self.model.fit(X)

        logger.debug('Computing the cluster names')
        self._compute_cluster_names(X)

    def predict(self, X: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Predict the labels of the documents

        Parameters
        ----------
        X : Union[List[str], pd.Series]
            The documents

        Returns
        -------
        np.ndarray
            The labels
        """
        return self.model.predict(X)

    def transform(self, X: Union[List[str], pd.Series], last_step: str | None = None, exclude_from: str | None = None) -> np.ndarray:
        """
        Transform the documents

        Parameters
        ----------
        X : Union[List[str], pd.Series]
            The documents
        last_step : str | None, optional
            The last step to use, by default None (use all steps)
        exclude_from : str | None, optional
            The step to exclude from, by default None (use all steps)

        Returns
        -------
        np.ndarray
            The transformed documents
        """
        # HACK: Find a way to do this in a more elegant way
        if last_step is None and exclude_from is None:
            return self.model.transform(X)
        else:
            for step in self.model.named_steps:
                if last_step is not None and last_step in step:
                    return self.model.named_steps[step].transform(X)

                if exclude_from is not None and exclude_from in step:
                    return X  # type: ignore

                X = self.model.named_steps[step].transform(X)

        return X  # type: ignore

    def save(self, name: str, overwrite: bool = False, save_only: Union[str, List[str]] | None = None) -> None:
        """
        Save the model to the models directory

        Parameters
        ----------
        name : str
            The name of the model
        overwrite : bool, optional
            Whether to overwrite the model if it already exists, by default False
        save_only : List[str] | None, optional
            The only parts to save, by default None (save everything)
        """
        # Set the name
        self.set_name(name)

        if save_only is None:
            save_only = ['model', 'config', 'cluster_names']

        if isinstance(save_only, str):
            save_only = [save_only]

        logger.debug('Checking model directory')
        save_to_dir = os.path.join(get_models_dir(), self._name)  # type: ignore
        os.makedirs(save_to_dir, exist_ok=overwrite)

        # Save the model
        if 'model' in save_only:
            if overwrite and os.path.exists(os.path.join(save_to_dir, 'model.joblib')):
                logger.debug('Removing the old model')
                os.remove(os.path.join(save_to_dir, 'model.joblib'))
            logger.debug('Saving the model')
            joblib.dump(self.model, os.path.join(save_to_dir, 'model.joblib'))

        # Save the config
        if 'config' in save_only:
            config = {
                'manager': 'SimpleSklearnPipelineManager',
                'random_state': self.random_state,
                'preprocessing_args': self.preprocessing_args
            }

            if overwrite and os.path.exists(os.path.join(save_to_dir, 'config.json')):
                logger.debug('Removing the old config')
                os.remove(os.path.join(save_to_dir, 'config.json'))
            logger.debug('Saving the config')
            with open(os.path.join(save_to_dir, 'config.json'), 'w') as f:
                json.dump(config, f)

        # Save the cluster names
        if 'cluster_names' in save_only:
            if overwrite and os.path.exists(os.path.join(save_to_dir, 'cluster_names.csv')):
                logger.debug('Removing the old cluster names')
                os.remove(os.path.join(save_to_dir, 'cluster_names.csv'))
            logger.debug('Saving the cluster names')
            self.cluster_names.to_csv(os.path.join(save_to_dir, 'cluster_names.csv'))

    @staticmethod
    def load(name: str) -> 'SimpleSklearnPipelineManager':
        """
        Load a model from the models directory

        Parameters
        ----------
        name : str
            The name of the model

        Returns
        -------
        SimpleSklearnPipelineManager
            The model manager
        """

        load_from_dir = os.path.join(get_models_dir(), name)

        # Load the model
        logger.debug('Loading the model')
        model = joblib.load(os.path.join(load_from_dir, 'model.joblib'))

        # Load the config
        logger.debug('Loading the config')
        with open(os.path.join(load_from_dir, 'config.json'), 'r') as f:
            config = json.load(f)

        # Create the model manager
        logger.debug('Creating the model manager')
        manager = SimpleSklearnPipelineManager(model, config['preprocessing_args'], config['random_state'])

        # Load the cluster names
        logger.debug('Loading the cluster names')
        manager.cluster_names = pd.read_csv(os.path.join(load_from_dir, 'cluster_names.csv'), index_col=0)

        # Set the name of the manager to the name of the model's directory
        manager.set_name(name)

        return manager

    def evaluate(self, X: Union[List[str], pd.Series], y: np.ndarray | None = None, scores: Dict[str, Callable[[Any, Any], float]] | None = None, transformed: bool = False) -> pd.DataFrame:
        """
        Evaluate the model by calculating the given scores

        Parameters
        ----------
        X : Union[List[str], pd.Series]
            The documents
        y : np.ndarray
            The labels
        scores : Dict[str, Callable[[Any, Any], float]] | None, optional
            The scores to compute, by default Silhouette, Calinski-Harabasz and Davies-Bouldin
        transformed : bool, optional
            Whether the data has already been transformed, by default False

        Returns
        -------
        pd.DataFrame
            The scores
        """
        if y is None:
            logger.debug('Predicting the labels')
            y = self.predict(X)

        if not transformed:
            logger.debug('Transforming the data')
            X = self.transform(X, exclude_from='clustering')

        return self._compute_scores(X, y, scores)

    def _preprocess(self, X: Union[List[str], Series]) -> Union[List[str], Series]:
        """
        Returns the unmodified input

        Parameters
        ----------
        X : Union[List[str], Series]
            The documents

        Returns
        -------
        Union[List[str], Series]
            The unmodified input
        """
        return X

    def _preprocess_single(self, x: str) -> str:
        """
        Returns the unmodified input

        Parameters
        ----------
        x : str
            The document

        Returns
        -------
        str
            The unmodified input
        """
        return x

    def _compute_cluster_names(self, X: Union[List[str], pd.Series]) -> None:
        """
        Compute the cluster names and store them in the cluster_names attribute

        Parameters
        ----------
        X : Union[List[str], pd.Series]
            The documents
        y : np.ndarray
            The labels
        """
        X_tfidf = self.transform(X, last_step='vectorize')

        cluster_names_dict = {}

        # Get the cluster words
        clustering_step = [step for step in self.model.named_steps.keys() if 'clustering' in step][0]
        labels = self.model.named_steps[clustering_step].labels_

        pbar = tqdm(np.unique(labels), desc='Computing cluster names', disable=not logger.isEnabledFor(logging.INFO))

        for cluster_id in pbar:
            cluster_mask = (labels == cluster_id)
            cluster_words = X_tfidf[cluster_mask].sum(axis=0).A1

            # Sort the words by their frequency
            indices_sorted = cluster_words.argsort()[::-1].tolist()

            # Get the top 10 words
            vectorize_step = [step for step in self.model.named_steps.keys() if 'vectorize' in step][0]
            vocab = self.model.named_steps[vectorize_step].get_feature_names_out()
            cluster_names_dict[cluster_id] = {'name': ', '.join(vocab[indices_sorted[:3]])}

        # Save the top 10 cluster words
        self.cluster_names = pd.DataFrame.from_dict(cluster_names_dict, orient='index')

    def generate_trends(self, query: str, resolution: str, source_index: str, distance: str = 'cosine', closest_n: int = 1, ignore_top_n_dates: int = 0, es: Elasticsearch | None = None) -> pd.DataFrame:
        """
        Generate trends for the given query

        Parameters
        ----------
        query : str
            The query to use
        resolution : str
            The resolution of the trends
        source_index : str
            The index containing the abstracts
        distance : str, optional
            The distance to use, by default 'cosine'
        closest_n : int, optional
            The number of closest clusters to use, by default 1
        ignore_top_n_dates : int, optional
            The number of biggest spikes to ignore, by default 0
        es : Elasticsearch | None, optional
            The Elasticsearch client, by default None

        Returns
        -------
        pd.DataFrame
            The trends
        """
        if es is None:
            es = get_es_client()

        # Get the query embedding
        query_embedding = self.transform([query], exclude_from='clustering')[0]

        # Get the cluster centers
        clustering_step = [step for step in self.model.named_steps.keys() if 'clustering' in step][0]
        cluster_centers = self.model.named_steps[clustering_step].cluster_centers_  # type: ignore

        match distance:
            case 'cosine':
                distances = 1 - cosine_similarity(cluster_centers, query_embedding.reshape(1, -1)).flatten()
            case 'euclidean':
                distances = np.linalg.norm(cluster_centers - query_embedding, axis=1)
            case _:
                logger.error(f'Invalid distance: {distance}')
                raise ValueError(f'Invalid distance: {distance}')

        # Sort the clusters by their distance to the query
        closest_cluster_labels = np.argsort(distances)[: closest_n]

        # Get the trends for each label
        trends_list = [self._get_label_trend(source_index, label, resolution, ignore_top_n_dates, es) for label in closest_cluster_labels]

        # Merge the trends
        trends = trends_list[0]
        for trend in trends_list[1:]:
            trends = trends.merge(trend, on='date', how='outer').fillna(0)

        # Sort by date
        trends = trends.sort_values('date')

        return trends


class SpacySklearnPipelineManager(ModelManager):
    def __init__(self, model: Pipeline, nlp: spacy.Language, preprocessing_args: Dict[str, bool] | None = None, random_state: int | None = None) -> None:
        """
        Initialize the model manager for TF-IDF-based pipelines with named entity recognition

        Parameters
        ----------
        model : Pipeline
            The model to use
        nlp : spacy.Language
            The spaCy model to use
        preprocessing_args : Dict[str, bool]
            The preprocessing arguments to use
        random_state : int | None, optional
            The random state to use, by default None
        """
        super().__init__(model, preprocessing_args, random_state)
        self.model = model
        self.nlp = nlp

    def fit(self, X: Union[List[str], pd.Series], preprocessed: bool = False) -> None:
        """
        Fit the model

        Parameters
        ----------
        X : Union[List[str], pd.Series]
            The documents
        preprocessed : bool, optional
            Whether the data has already been preprocessed, by default False
        """
        if not preprocessed:
            logger.debug('Preprocessing the data')
            X = self._preprocess(X)

        logger.debug('Fitting the model')
        self.model.fit(X)

        logger.debug('Computing the cluster names')
        self._compute_cluster_names(X, preproccessed=True)

    def predict(self, X: Union[List[str], pd.Series], preprocessed: bool = False) -> np.ndarray:
        """
        Predict the labels of the documents

        Parameters
        ----------
        X : Union[List[str], pd.Series]
            The documents
        preprocessed : bool, optional
            Whether the data has already been preprocessed, by default False

        Returns
        -------
        np.ndarray
            The labels
        """
        if not preprocessed:
            logger.debug('Preprocessing the data')
            X = self._preprocess(X)

        return self.model.predict(X)

    def transform(self, X: Union[List[str], pd.Series], last_step: str | None = None, exclude_from: str | None = None, preprocessed: bool = False) -> np.ndarray:
        """
        Transform the documents

        Parameters
        ----------
        X : Union[List[str], pd.Series]
            The documents
        last_step : str | None, optional
            The name of the last step to include, by default None
        exclude_last_steps : int | None, optional
            The number of steps to exclude from the end of the pipeline, by default None
            This is meant for cases like KMeans, which has a transform method, but which operates in an unusual way
        preprocessed : bool, optional
            Whether the data has already been preprocessed, by default False

        Returns
        -------
        np.ndarray
            The transformed documents
        """
        if not preprocessed:
            logger.debug('Preprocessing the data')
            X = self._preprocess(X)

        # HACK: Find a way to do this in a more elegant way
        if last_step is None and exclude_from is None:
            return self.model.transform(X)
        else:
            for step in self.model.named_steps:
                if last_step is not None and last_step in step:
                    return self.model.named_steps[step].transform(X)

                if exclude_from is not None and exclude_from in step:
                    return X  # type: ignore

                X = self.model.named_steps[step].transform(X)

        return X  # type: ignore

    def save(self, name: str, overwrite: bool = False, save_only: Union[str, List[str]] | None = None, spacy_config_only: bool = False) -> None:
        """
        Save the model to the models directory

        Parameters
        ----------
        name : str
            The name of the model
        overwrite : bool, optional
            Whether to overwrite the model if it already exists, by default False
        save_only : List[str] | None, optional
            The only parts to save, by default None (save everything)
        spacy_config_only : bool, optional
            (Experimental) Whether to save only the spacy config instead of the whole model, by default False
        """
        # Set the name
        self.set_name(name)

        if save_only is None:
            save_only = ['model', 'spacy', 'config', 'cluster_names']

        if isinstance(save_only, str):
            save_only = [save_only]

        logger.debug('Checking model directory')
        save_to_dir = os.path.join(get_models_dir(), self._name)  # type: ignore
        os.makedirs(save_to_dir, exist_ok=overwrite)

        # Save the model
        if 'model' in save_only:
            if overwrite and os.path.exists(os.path.join(save_to_dir, 'model.joblib')):
                logger.debug('Removing the old model')
                os.remove(os.path.join(save_to_dir, 'model.joblib'))
            logger.debug('Saving the model')
            joblib.dump(self.model, os.path.join(save_to_dir, 'model.joblib'))

        # Save the config
        if 'config' in save_only:
            config = {
                'manager': 'SpacySklearnPipelineManager',
                'random_state': self.random_state,
                'preprocessing_args': self.preprocessing_args
            }

            if overwrite and os.path.exists(os.path.join(save_to_dir, 'config.json')):
                logger.debug('Removing the old config')
                os.remove(os.path.join(save_to_dir, 'config.json'))
            logger.debug('Saving the config')
            with open(os.path.join(save_to_dir, 'config.json'), 'w') as f:
                json.dump(config, f)

        # Save the spacy model
        if 'spacy' in save_only:
            if spacy_config_only:
                if overwrite and os.path.exists(os.path.join(save_to_dir, 'spacy.json')):
                    logger.debug('Removing the old spacy config')
                    os.remove(os.path.join(save_to_dir, 'spacy.json'))

                logger.debug('Saving the spacy config')
                with open(os.path.join(save_to_dir, 'spacy.json'), 'w') as f:
                    json.dump(self.nlp.meta, f)
            else:
                if overwrite and os.path.exists(os.path.join(save_to_dir, 'spacy.nlp')):
                    logger.debug('Removing the old spacy model')
                    os.remove(os.path.join(save_to_dir, 'spacy.nlp'))

                logger.debug('Saving the spacy model')
                self.nlp.to_disk(os.path.join(save_to_dir, 'spacy.nlp'))

        # Save the cluster names
        if 'cluster_names' in save_only:
            if overwrite and os.path.exists(os.path.join(save_to_dir, 'cluster_names.csv')):
                logger.debug('Removing the old cluster names')
                os.remove(os.path.join(save_to_dir, 'cluster_names.csv'))
            logger.debug('Saving the cluster names')
            self.cluster_names.to_csv(os.path.join(save_to_dir, 'cluster_names.csv'))

    @staticmethod
    def load(name: str) -> 'SpacySklearnPipelineManager':
        """
        Load a model from the models directory

        Parameters
        ----------
        name : str
            The name of the model

        Returns
        -------
        SpacySklearnPipelineManager
            The model manager
        """
        load_from_dir = os.path.join(get_models_dir(), name)

        # Load the model
        logger.debug('Loading the model')
        model = joblib.load(os.path.join(load_from_dir, 'model.joblib'))

        # Load the config
        logger.debug('Loading the config')
        with open(os.path.join(load_from_dir, 'config.json'), 'r') as f:
            config = json.load(f)

        # Find out if the spacy model is saved as a config or as a whole model
        if os.path.exists(os.path.join(load_from_dir, 'spacy.nlp')):
            # Load the spacy model
            logger.debug('Loading the spacy model')
            nlp = spacy.load(os.path.join(load_from_dir, 'spacy.nlp'))
        else:
            # Load the spacy config
            logger.debug('Loading the spacy config')
            with open(os.path.join(load_from_dir, 'spacy.json'), 'r') as f:
                spacy_config = json.load(f)

            # Load the spacy model
            logger.debug('Loading the spacy model')
            nlp = spacy.load(spacy_config['name'], disable=spacy_config['disabled'])

        # Create the model manager
        logger.debug('Creating the model manager')
        manager = SpacySklearnPipelineManager(model, nlp, config['preprocessing_args'], config['random_state'])

        # Load the cluster names
        logger.debug('Loading the cluster names')
        manager.cluster_names = pd.read_csv(os.path.join(load_from_dir, 'cluster_names.csv'), index_col=0)

        # Set the name of the manager to the name of the model's directory
        manager.set_name(name)

        return manager

    def evaluate(self, X: Union[List[str], pd.Series], y: np.ndarray | None = None, scores: Dict[str, Callable[[Any, Any], float]] | None = None, transformed: bool = False, preprocessed: bool = False) -> pd.DataFrame:
        """
        Evaluate the model by calculating the given scores

        Parameters
        ----------
        X : Union[List[str], pd.Series]
            The documents
        y : np.ndarray | None, optional
            The labels
        scores : Dict[str, Callable[[Any, Any], float]] | None, optional
            The scores to compute, by default Silhouette, Calinski-Harabasz and Davies-Bouldin
        transformed : bool, optional
            Whether the data has already been transformed, by default False
        preprocessed : bool, optional
            Whether the data has already been preprocessed, by default False

        Returns
        -------
        pd.DataFrame
            The scores
        """
        if not preprocessed:
            logger.debug('Preprocessing the data')
            X = self._preprocess(X)

        if y is None:
            logger.debug('Predicting the labels')
            y = self.predict(X, preprocessed=True)

        if not transformed:
            logger.debug('Transforming the data')
            X = self.transform(X, preprocessed=True, exclude_from='clustering')

        return self._compute_scores(X, y, scores)

    def _preprocess(self, X: Union[List[str], pd.Series]) -> Union[List[str], pd.Series]:
        """
        Preprocess a batch of documents

        Parameters
        ----------
        X : Union[List[str], pd.Series]
            The documents

        Returns
        -------
        Union[List[str], pd.Series]
            The preprocessed documents
        """
        pbar = tqdm(X, desc='Preprocessing', disable=not logger.isEnabledFor(logging.INFO))
        return [self._preprocess_single(x) for x in pbar]

    def _preprocess_single(self, x: str) -> str:
        """
        Preprocess a single document

        Parameters
        ----------
        x : str
            The document

        Returns
        -------
        str
            The preprocessed document
        """
        doc = self.nlp(x)
        return ' '.join([ent.text for ent in doc.ents])

    def _compute_cluster_names(self, X: Union[List[str], pd.Series], preproccessed: bool = False) -> None:
        """
        Compute the cluster names

        Parameters
        ----------
        X : Union[List[str], pd.Series]
            The documents
        y : np.ndarray
            The labels
        preproccessed : bool, optional
            Whether the data has already been preprocessed, by default False

        Returns
        -------
        Dict[int, str]
            The cluster names
        """
        if not preproccessed:
            logger.debug('Preprocessing the data')
            X = self._preprocess(X)

        X_tfidf = self.transform(X, last_step='vectorize')

        cluster_names_dict = {}

        # Get the cluster words
        clustering_step = [step for step in self.model.named_steps.keys() if 'clustering' in step][0]
        labels = self.model.named_steps[clustering_step].labels_

        pbar = tqdm(np.unique(labels), desc='Computing cluster names', disable=not logger.isEnabledFor(logging.INFO))

        for cluster_id in pbar:
            cluster_mask = (labels == cluster_id)
            cluster_words = X_tfidf[cluster_mask].sum(axis=0).A1

            # Sort the words by their frequency
            indices_sorted = cluster_words.argsort()[::-1].tolist()

            # Get the top 10 words
            vectorize_step = [step for step in self.model.named_steps.keys() if 'vectorize' in step][0]
            vocab = self.model.named_steps[vectorize_step].get_feature_names_out()
            cluster_names_dict[cluster_id] = {'name': ', '.join(vocab[indices_sorted[:3]])}

        # Save the top 10 cluster words
        self.cluster_names = pd.DataFrame.from_dict(cluster_names_dict, orient='index')

    def generate_trends(self, query: str, resolution: str, source_index: str, distance: str = 'cosine', closest_n: int = 1, ignore_top_n_dates: int = 0, es: Elasticsearch | None = None) -> pd.DataFrame:
        """
        Generate trends for the given query

        Parameters
        ----------
        query : str
            The query to use
        resolution : str
            The resolution of the trends
        source_index : str
            The index containing the abstracts
        distance : str, optional
            The distance to use, by default 'cosine'
        closest_n : int, optional
            The number of closest clusters to use, by default 1
        ignore_top_n_dates : int, optional
            The number of biggest spikes to ignore, by default 0
        es : Elasticsearch | None, optional
            The Elasticsearch client, by default None

        Returns
        -------
        pd.DataFrame
            The trends
        """
        if es is None:
            es = get_es_client()

        # Preprocess the query
        query = self._preprocess_single(query)

        # Get the query embedding
        query_embedding = self.transform([query], exclude_from='clustering')[0]

        # Get the cluster centers
        clustering_step = [step for step in self.model.named_steps.keys() if 'clustering' in step][0]
        cluster_centers = self.model.named_steps[clustering_step].cluster_centers_  # type: ignore

        match distance:
            case 'cosine':
                distances = 1 - cosine_similarity(cluster_centers, query_embedding.reshape(1, -1)).flatten()
            case 'euclidean':
                distances = np.linalg.norm(cluster_centers - query_embedding, axis=1)
            case _:
                logger.error(f'Invalid distance: {distance}')
                raise ValueError(f'Invalid distance: {distance}')

        # Sort the clusters by their distance to the query
        closest_cluster_labels = np.argsort(distances)[: closest_n]

        # Get the trends for each label
        trends_list = [self._get_label_trend(source_index, label, resolution, ignore_top_n_dates, es) for label in closest_cluster_labels]

        # Merge the trends
        trends = trends_list[0]
        for trend in trends_list[1:]:
            trends = trends.merge(trend, on='date', how='outer').fillna(0)

        # Sort by date
        trends = trends.sort_values('date')

        return trends


class BERTopicManager(ModelManager):
    def __init__(self, model: BERTopic, preprocessing_args: Dict[str, bool] | None = None, random_state: int | None = None) -> None:
        """
        Initialize the model manager for BERTopic

        Parameters
        ----------
        model : BERTopic
            The model to use
        preprocessing_args : Dict[str, bool] | None, optional
            The preprocessing arguments to use
        random_state : int | None, optional
            The random state to use, by default None
        """
        super().__init__(model, preprocessing_args, random_state)
        self.model = model

    def fit(self, X: Union[List[str], Series]) -> None:
        """
        Fit the model

        Parameters
        ----------
        X : Union[List[str], Series]
            The documents
        """
        logger.debug('Fitting the model')
        self.model.fit(X)

        logger.debug('Computing the cluster names')
        self._compute_cluster_names(X)

    def predict(self, X: Union[List[str], Series]) -> np.ndarray:
        """
        Predict the labels of the documents

        Parameters
        ----------
        X : Union[List[str], Series]
            The documents

        Returns
        -------
        np.ndarray
            The labels
        """
        logger.debug('Predicting the labels')
        topics, probs = self.model.transform(X)

        return np.array(topics)

    def transform(self, X: Union[List[str], Series], last_step: str | None = None, exclude_from: str | None = None) -> np.ndarray:
        """
        Transform the documents.

        Parameters
        ----------
        X : Union[List[str], Series]
            The documents
        last_step : str, optional
            ignored
        exclude_from : str, optional
            ignored

        Returns
        -------
        np.ndarray
            The transformed documents
        """
        logger.debug('Extracting embeddings')
        embeddings = self.model._extract_embeddings(X, method="document")

        logger.debug('Reducing the dimensionality')
        reduced_embeddings = np.nan_to_num(self.model.umap_model.transform(embeddings))

        return reduced_embeddings

    def save(self, name: str, overwrite: bool = False, save_only: Union[str, List[str]] | None = None) -> None:
        """
        Save the model to the models directory

        Parameters
        ----------
        name : str
            The name of the model
        overwrite : bool, optional
            Whether to overwrite the model if it already exists, by default False
        save_only : List[str] | None, optional
            The only parts to save, by default None (save everything)
        """
        # Set the name
        self.set_name(name)

        if save_only is None:
            save_only = ['model', 'config', 'predictions', 'cluster_names']

        if isinstance(save_only, str):
            save_only = [save_only]

        logger.debug('Checking model directory')
        save_to_dir = os.path.join(get_models_dir(), self._name)  # type: ignore
        os.makedirs(save_to_dir, exist_ok=overwrite)

        # Save the model
        if 'model' in save_only:
            if overwrite and os.path.exists(os.path.join(save_to_dir, 'model.joblib')):
                logger.debug('Removing the old model')
                os.remove(os.path.join(save_to_dir, 'model.joblib'))
            logger.debug('Saving the model')
            self.model.save(os.path.join(save_to_dir, 'model.joblib'))

        # Save the config
        if 'config' in save_only:
            config = {
                'manager': 'BERTopicManager',
                'random_state': self.random_state,
                'preprocessing_args': self.preprocessing_args
            }

            if overwrite and os.path.exists(os.path.join(save_to_dir, 'config.json')):
                logger.debug('Removing the old config')
                os.remove(os.path.join(save_to_dir, 'config.json'))
            logger.debug('Saving the config')
            with open(os.path.join(save_to_dir, 'config.json'), 'w') as f:
                json.dump(config, f)

        # Save the cluster names
        if 'cluster_names' in save_only:
            if overwrite and os.path.exists(os.path.join(save_to_dir, 'cluster_names.csv')):
                logger.debug('Removing the old cluster names')
                os.remove(os.path.join(save_to_dir, 'cluster_names.csv'))
            logger.debug('Saving the cluster names')
            self.cluster_names.to_csv(os.path.join(save_to_dir, 'cluster_names.csv'))

    @staticmethod
    def load(name: str) -> 'BERTopicManager':
        """
        Load a model from the models directory

        Parameters
        ----------
        name : str
            The name of the model

        Returns
        -------
        BERTopicManager
            The model manager
        """
        load_from_dir = os.path.join(get_models_dir(), name)

        # Load the model
        logger.debug('Loading the model')
        model = BERTopic.load(os.path.join(load_from_dir, 'model.joblib'))

        model.verbose = logger.isEnabledFor(logging.DEBUG)

        # Load the config
        logger.debug('Loading the config')
        with open(os.path.join(load_from_dir, 'config.json'), 'r') as f:
            config = json.load(f)

        # Create the model manager
        logger.debug('Creating the model manager')
        manager = BERTopicManager(model, config['preprocessing_args'], config['random_state'])

        # Load the cluster names
        logger.debug('Loading the cluster names')
        manager.cluster_names = pd.read_csv(os.path.join(load_from_dir, 'cluster_names.csv'), index_col=0)

        # Set the name of the manager to the name of the model's directory
        manager.set_name(name)

        return manager

    def evaluate(self, X: Union[List[str], Series], y: np.ndarray | None = None, scores: Dict[str, Callable[[Any, Any], float]] | None = None, transformed: bool = False) -> pd.DataFrame:
        """
        Evaluate the model by calculating the given scores

        Parameters
        ----------
        X : Union[List[str], Series]
            The documents
        y : np.ndarray
            The labels
        scores : Dict[str, Callable[[Any, Any], float]] | None, optional
            The scores to compute, by default Silhouette, Calinski-Harabasz and Davies-Bouldin
        transformed : bool, optional
            Whether the data has already been transformed, by default False

        Returns
        -------
        pd.DataFrame
            The scores
        """
        if y is None:
            logger.debug('Predicting the labels')
            y = self.predict(X)

        if not transformed:
            logger.debug('Transforming the data')
            X = self.transform(X)

        return self._compute_scores(X, y, scores)

    def _preprocess(self, X: Union[List[str], Series]) -> Union[List[str], Series]:
        """
        Returns the unmodified input

        Parameters
        ----------
        X : Union[List[str], Series]
            The documents

        Returns
        -------
        Union[List[str], Series]
            The unmodified input
        """
        return X

    def _preprocess_single(self, x: str) -> str:
        """
        Returns the unmodified input

        Parameters
        ----------
        x : str
            The document

        Returns
        -------
        str
            The unmodified input
        """
        return x

    def _compute_cluster_names(self, X: Union[List[str], Series]) -> None:
        """
        Compute the cluster names

        Parameters
        ----------
        X : Union[List[str], Series]
            The documents
        y : np.ndarray
            The labels

        Returns
        -------
        Dict[int, str]
            The cluster names
        """
        topic_info = self.model.get_topic_info()

        # Save the top 10 cluster words
        self.cluster_names = pd.DataFrame(topic_info).set_index('Topic').drop(columns=['Count']).rename(columns={'Name': 'name'})

    def generate_trends(self, query: str, resolution: str, source_index: str, distance: str = 'cosine', closest_n: int = 1, ignore_top_n_dates: int = 0, es: Elasticsearch | None = None) -> pd.DataFrame:
        """
        Generate trends for the given query

        Parameters
        ----------
        query : str
            The query to use
        resolution : str
            The temporal resolution
        source_index : str
            The index containing the abstracts
        distance : str, optional
            ignored
        closest_n : int, optional
            The number of closest clusters to generate trends for, by default 1
        ignore_top_n_dates : int, optional
            The number of biggest spikes to ignore, by default 0
        es : Elasticsearch | None, optional
            The Elasticsearch client, by default None

        Returns
        -------
        pd.DataFrame
            The trends
        """
        if es is None:
            es = get_es_client()

        topics, probs = self.model.find_topics(query, top_n=closest_n + 1)

        # Sort the topics list by probability
        topics = [topic for _, topic in sorted(zip(probs, topics), reverse=True) if not topic == -1]

        # Get the trends for each label
        trends_list = [self._get_label_trend(source_index, label, resolution, ignore_top_n_dates, es) for label in topics[:closest_n]]

        # Merge the trends
        trends = trends_list[0]
        for trend in trends_list[1:]:
            trends = trends.merge(trend, on='date', how='outer').fillna(0)

        # Sort by date
        trends = trends.sort_values('date')

        return trends


class Top2VecManager(ModelManager):

    def __init__(self, model: Top2Vec, preprocessing_args: Dict[str, bool] | None = None, random_state: int | None = None) -> None:
        super().__init__(model, preprocessing_args, random_state)
        self.model = model
        self.umap_args = {
            'n_neighbors': 15,
            'n_components': 5,
            'metric': 'cosine'
        }

    def fit(self, X: List[str] | pd.Series) -> None:
        logger.debug('Fitting the model')
        if type(X) == pd.Series:
            X = X.tolist()
        self.model = Top2Vec(documents=X,
                             verbose=True,
                             embedding_model='doc2vec',
                             workers=os.cpu_count())

        logger.debug('Computing the cluster names')
        self._compute_cluster_names(X)

    def predict(self, X: List[str] | pd.Series) -> np.ndarray:
        logger.debug('Predicting the labels')

        if type(X) == pd.Series:
            X = X.tolist()

        if type(X) == str:
            X = [X]

        y = np.zeros(len(X))
        for i, x in enumerate(X):
            _, _, _, topic_nums = self.model.query_topics(query=x, num_topics=1)
            y[i] = topic_nums[0]

        return y.astype(int)

    def transform(self, X: List[str] | pd.Series, last_step: str | None = None, exclude_from: str | None = None, tokenizer: Callable | None = None) -> np.ndarray:
        if type(X) == pd.Series:
            X = X.tolist()

        if type(X) == str:
            X = [X]

        X_transformed = []

        logger.debug('Transforming the data')
        # Copied from top2vec.Top2Vec.query_topics without the predictions step:
        for x in tqdm(X):
            self.model._validate_query(x)

            if self.model.embedding_model != "doc2vec":
                query_vec = self.model._embed_query(x)

            else:
                if tokenizer is None:
                    tokenizer = default_tokenizer

                tokenized_query = tokenizer(x)
                query_vec = self.model.model.infer_vector(doc_words=tokenized_query, alpha=0.025, min_alpha=0.01, epochs=100)

            X_transformed.append(query_vec)

        return np.array(X_transformed)

    def save(self, name: str, overwrite: bool = False, save_only: str | List[str] | None = None) -> None:
        self.set_name(name)

        if save_only is None:
            save_only = ['model', 'config', 'predictions', 'cluster_names']

        if isinstance(save_only, str):
            save_only = [save_only]

        logger.debug('Checking model directory')
        save_to_dir = os.path.join(get_models_dir(), self._name)  # type: ignore
        os.makedirs(save_to_dir, exist_ok=True)

        if 'model' in save_only:
            if overwrite and os.path.exists(os.path.join(save_to_dir, 'model')):
                logger.debug('Removing the old model')
                os.remove(os.path.join(save_to_dir, 'model'))
            logger.debug('Saving model')
            self.model.save(os.path.join(save_to_dir, 'model.joblib'))

        # save config
        if 'config' in save_only:
            config = {
                'manager': 'Top2VecManager',
                'random_state': self.random_state,
                'preprocessing_args': self.preprocessing_args
            }

            if overwrite and os.path.exists(os.path.join(save_to_dir, 'config.json')):
                logger.debug('Removing the old config')
                os.remove(os.path.join(save_to_dir, 'config.json'))
            logger.debug('Saving config')
            with open(os.path.join(save_to_dir, 'config.json'), 'w') as f:
                json.dump(config, f)

        # save cluster names
        if 'cluster_names' in save_only:
            if overwrite and os.path.exists(os.path.join(save_to_dir, 'cluster_names.csv')):
                logger.debug('Removing the old cluster names')
                os.remove(os.path.join(save_to_dir, 'cluster_names.csv'))
            logger.debug('Saving cluster names')
            self.cluster_names.to_csv(os.path.join(save_to_dir, 'cluster_names.csv'))

    @staticmethod
    def load(name: str) -> Top2Vec:
        load_from_dir = os.path.join(get_models_dir(), name)

        logger.debug('Loading the model')
        model = Top2Vec.load(os.path.join(load_from_dir, 'model.joblib'))

        model.verbose = logger.isEnabledFor(logging.DEBUG)

        logger.debug('Loading the config')
        with open(os.path.join(load_from_dir, 'config.json'), 'r') as f:
            config = json.load(f)

        logger.debug('Creating the model manager')
        manager = Top2VecManager(model, config['preprocessing_args'], config['random_state'])

        # Load the cluster names
        logger.debug('Loading the cluster names')
        manager.cluster_names = pd.read_csv(os.path.join(load_from_dir, 'cluster_names.csv'), index_col=0)

        manager.set_name(name)

        return manager

    def evaluate(self, X: List[str] | pd.Series, y: np.ndarray | None = None,
                 scores: Dict[str, Callable[[Any, Any], float]] = None, transformed: bool = False) -> pd.DataFrame:
        if y is None:
            logger.debug('Predicting the labels')
            y = self.predict(X)

        if not transformed:
            logger.debug('Transforming the data')
            X = self.transform(X)

        return self._compute_scores(X, y, scores)

    def generate_trends(self, query: str, resolution: str, source_index: str, distance: str | None = 'cosine',
                        closest_n: int = 1, ignore_top_n_dates: int = 0, es: Elasticsearch = None) -> pd.DataFrame:
        if es is None:
            es = get_es_client()

        _, _, _, topic_nums = self.model.query_topics(query=query, num_topics=closest_n + 1)

        trends_list = [self._get_label_trend(source_index, label, resolution, ignore_top_n_dates, es) for label in topic_nums[:closest_n]]

        # Merge the trends
        trends = trends_list[0]
        for trend in trends_list[1:]:
            trends = trends.merge(trend, on='date', how='outer').fillna(0)

        # Sort by date
        trends = trends.sort_values(by='date')

        return trends

    def _compute_cluster_names(self, X: List[str] | pd.Series) -> None:
        topic_words, _, topic_nums = self.model.get_topics()

        self.cluster_names = pd.DataFrame({'Topic': topic_nums, 'name': ['_'.join(row) for row in topic_words[:, :5]]}).set_index('Topic')

    def _preprocess(self, X: List[str] | pd.Series) -> List[str] | pd.Series:
        return X

    def _preprocess_single(self, x: str) -> str:
        return x
