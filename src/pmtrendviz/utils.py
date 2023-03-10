import atexit
import logging
import os
import random
import shutil
import string
import time
import urllib.request
from typing import List

import pandas as pd
import platformdirs
from dotenv import dotenv_values
from pandas import DataFrame, to_datetime

from elasticsearch import Elasticsearch

logger = logging.getLogger('pmtrendviz.utils')


def get_es_client(user: str = 'elastic', pw: str | None = None, port: int = 9200, hosts: List[str] | None = None, scheme: str = 'http', environment_variables_path: str | None = None) -> Elasticsearch:
    """
    Get a client for Elasticsearch. Uses python-dotenv to read the environment variables from the file specified by the environment_variables_path argument.

    Parameters
    ----------
    user : str, optional
        The user name, by default 'elastic'
    pw : str | None, optional
        The password, by default None
    port : int, optional
        The port, by default 9200
    hosts : List[str] | None, optional
        The hosts, by default ['localhost']
    scheme : str, optional
        The scheme, by default 'http'
    environment_variables_path : str | None, optional
        The path to the elasticsearch environment variables file, by default None

    Returns
    -------
    Elasticsearch
        The Elasticsearch client
    """
    hosts = ['localhost'] if hosts is None else hosts
    if pw is None:
        if environment_variables_path is None:
            conf_file = os.path.join(platformdirs.user_config_dir('pmtrendviz'), 'es.env')
            if not os.path.exists(conf_file):
                conf_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'elasticsearch', 'es.env'))
            conf = dotenv_values(conf_file)
        else:
            conf = dotenv_values(environment_variables_path)
        pw = conf['ELASTIC_PASSWORD']

    return Elasticsearch(
        hosts,
        http_auth=(user, pw),
        scheme=scheme,
        port=port)


class UserAbort(Exception):
    """Exception raised when the user aborts the program."""

    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return f"User aborted the program at: {self.message}."


class MaxRetriesExceeded(Exception):
    """Exception raised when the maximum number of retries is exceeded."""

    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return f"Maximum number of retries exceeded: {self.message}."


def get_models_dir() -> str:
    """
    Get the path to the models directory

    Returns
    -------
    str
        The path to the models directory
    """
    MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models', 'saved')

    os.makedirs(MODELS_DIR, exist_ok=True)

    return MODELS_DIR


def get_cache_dir() -> str:
    """
    Get the path to the cache directory

    Returns
    -------
    str
        The path to the cache directory
    """

    CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')

    os.makedirs(CACHE_DIR, exist_ok=True)

    return CACHE_DIR


def get_data_dir() -> str:
    """
    Get the path to the cache directory

    Returns
    -------
    str
        The path to the cache directory
    """

    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

    os.makedirs(DATA_DIR, exist_ok=True)

    return DATA_DIR


def download_file(url: str, filepath: str, max_retries: int = 2, backoff_factor: float = 1.0, overwrite: bool = False) -> None:
    """
    Download a file from a given URL to a given filename.

    Parameters
    ----------
    url : str
        The URL to download the file from
    filepath : str
        The filepath to save the file to
    max_retries : int, optional
        The maximum number of retries, by default 2
    backoff_factor : float, optional
        The backoff factor, by default 1.0
    overwrite : bool, optional
        Whether to overwrite the file if it already exists, by default False
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, but is {max_retries}")

    if backoff_factor < 0:
        raise ValueError(f"backoff_factor must be >= 0, but is {backoff_factor}")

    if os.path.exists(filepath):
        if not overwrite:
            logger.debug(f"File {filepath} already exists. Skipping download.")
            return
        else:
            logger.debug(f"File {filepath} already exists. Deleting old file")
            os.remove(filepath)

    logger.debug(f"Downloading file from {url} to {filepath}")

    # Initialize variables
    retries = -1
    errors = []

    # Create a temporary file to download the file to
    # Since there may already be a file with the same name, we need to make sure that the temporary file has a unique name
    random_hash = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    filepath_temp = os.path.join(os.path.dirname(filepath), f"{random_hash}.{os.path.basename(filepath)}.tmp")

    # Make sure that the temporary file is deleted when the program exits
    atexit.register(lambda f: os.remove(f) if os.path.exists(f) else None, filepath_temp)

    while True:
        try:
            # Download the file
            urllib.request.urlretrieve(url, filepath_temp)
            break

        except Exception as e:
            errors.append(e)

            # Delete the temporary file if it still exists
            if os.path.exists(filepath_temp):
                os.remove(filepath_temp)

            retries += 1
            logger.warning(f"Error downloading file from {url} to {filepath}, try {retries}/{max_retries}, retrying in {backoff_factor * (2 ** retries)} seconds: {e}")
            if retries >= max_retries:
                break

            # Wait before retrying
            time.sleep(backoff_factor * (2 ** retries))

    if retries < max_retries:
        # On success, rename the temporary file to the final file
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(filepath_temp, filepath)
    else:
        # On failure, delete the temporary file and raise an exception
        if os.path.exists(filepath_temp):
            os.remove(filepath_temp)
        logger.error(f"Could not download file from {url} to {filepath} after {max_retries} retries.")
        raise MaxRetriesExceeded(f"Could not download file from {url} to {filepath} after {max_retries} retries: {errors}")


class ImportOptimizer:
    """
    Class to optimize the import of files into Elasticsearch by keeping track of which files have already been imported.
    """
    def __init__(self, index: str):
        """
        Parameters
        ----------
        index : str
            The index to which the files will be imported
        """
        self.index = index
        logger.debug(f"Initializing ImportOptimizer for index {index}")

        # If the index does not exist, but the file with the imported files does, delete the file
        es = get_es_client()

        # Check if the index exists and if it is empty
        if not es.indices.exists(index=index) or int(es.cat.count(index=index, params={"format": "json"})[0]["count"]) == 0:  # type: ignore
            logger.info(f"Index {index} does not exist. Deleting outdated imported files file.")
            self.reset_imported_files()

    def register_imported_file(self, filename: str) -> None:
        """
        Mark a file as imported by adding it to the file that lists all imported files of the index.

        Parameters
        ----------
        filename : str
            The filename
        """
        filename = os.path.basename(filename).split(".")[0]

        index_file = os.path.join(get_data_dir(), "imported_files", f"{self.index}.txt")

        if not os.path.exists(index_file):
            # Create the file
            logger.debug(f"Creating imported files file {index_file}")
            imported_files = set()
            os.makedirs(os.path.dirname(index_file), exist_ok=True)
        else:
            # Read the file
            logger.debug(f"Reading imported files file {index_file}")
            with open(index_file, "r") as f:
                imported_files = set([f.strip() for f in f.readlines()])

        # Add the filename to the set
        logger.debug(f"Adding filename {filename} to imported files file {index_file}")
        imported_files.add(filename)

        # Write the file
        logger.debug(f"Writing imported files file {index_file}")
        with open(index_file, "w") as f:
            for filename in imported_files:
                f.write(f"{filename}\n")

    def filter_registered_files(self, filenames: List[str]) -> List[str]:
        """
        Check if a file has already been imported to the index.

        Parameters
        ----------
        filenames : List[str]
            The filenames

        Returns
        -------
        List[str]
            The filenames that have not been imported yet
        """
        index_file = os.path.join(get_data_dir(), "imported_files", f"{self.index}.txt")

        # If the file does not exist, return all filenames
        if not os.path.exists(index_file):
            logger.debug(f"File with list of imported files {index_file} does not exist. Returning all filenames.")
            return filenames

        # Read the file with the imported files
        logger.debug(f"Reading imported files file {index_file}")
        with open(index_file, "r") as f:
            imported_files = set([f.strip() for f in f.readlines()])

        # Filter the filenames
        return [filename for filename in filenames if os.path.basename(filename).split(".")[0] not in imported_files]

    def reset_imported_files(self) -> None:
        """
        Reset (remove) the file that lists all imported files of the index.
        """
        index_file = os.path.join(get_data_dir(), "imported_files", f"{self.index}.txt")

        os.makedirs(os.path.dirname(index_file), exist_ok=True)

        # Delete the file
        logger.debug(f"Deleting imported files file {index_file}")
        if os.path.exists(index_file):
            os.remove(index_file)


def get_time_to_complete_offset(index: str, es: Elasticsearch | None = None) -> int:
    """
    Calculates the difference between the date completed and the article date and returns the mean offset in days.

    Parameters
    ----------
    index : str
        The index to search
    es : Elasticsearch | None, optional
        The Elasticsearch client, by default None

    Returns
    -------
    int
        The mean offset in days
    """
    if es is None:
        es = get_es_client()

    mean_offset_milliseconds = es.search(index=index, size=0, aggs={
        'mean_offset': {
            'avg': {
                'script': "return (doc['DATE_COMPLETED'].value.toInstant().toEpochMilli() - doc['ARTICLE.ARTICLE_DATE'].value.toInstant().toEpochMilli())"
            }
        }
    }, query={  # Only where the ARTICLE field has an ARTICLE_DATE field and the DATE_COMPLETED field is not null
        'bool': {
            'must': [
                {'exists': {'field': 'ARTICLE.ARTICLE_DATE'}},
                {'exists': {'field': 'DATE_COMPLETED'}}
            ]
        }
    })['aggregations']['mean_offset']['value']

    if mean_offset_milliseconds is None:
        return 0

    # Convert the mean offset into days
    mean_offset_days = round(mean_offset_milliseconds / (1000 * 60 * 60 * 24))

    return mean_offset_days


def get_articles_per_date(index: str, date_field: str = 'DATE_COMPLETED', es: Elasticsearch = None) -> DataFrame:
    """
    Get the number of articles published per day from Elasticsearch by means of a date histogram aggregation.

    Parameters
    ----------
    index : str
        The index to search
    date_field : ['DATE_COMPLETED', 'ARTICLE.ARTICLE_DATE', 'DATE_REVISED'], optional
        The field to use for the date, by default 'DATE_COMPLETED'
    es : Elasticsearch, optional
        The Elasticsearch client, by default None

    Returns
    -------
    pd.DataFrame
        A dataframe with
        - the number of articles published per day and
        - the rank of the day in terms of the number of articles published
    """
    if es is None:
        es = get_es_client()

    articles_per_date = DataFrame(es.search(index=index, size=0, aggs={
        'articles_per_date': {
            'date_histogram': {
                'field': date_field,
                'calendar_interval': 'day',
                'format': 'yyyy-MM-dd'
            }
        }
    })['aggregations']['articles_per_date']['buckets'])

    if len(articles_per_date) == 0:
        return pd.DataFrame(columns=['date', 'articles', 'rank'])

    articles_per_date = articles_per_date.drop(columns='key').rename(columns={'key_as_string': 'date', 'doc_count': 'articles'})

    # Convert the date column to datetime
    articles_per_date['date'] = to_datetime(articles_per_date['date'])

    # Rank the articles per date
    articles_per_date['rank'] = articles_per_date['articles'].rank(method='first', ascending=False).astype(int)

    return articles_per_date


def extract_dates_from_document(doc: dict, parse_dates: bool = False) -> dict:
    """
    Extract the dates from the document

    Parameters
    ----------
    doc : dict
        The elasticsearch document
    parse_dates : bool, optional
        Whether to parse the dates, by default False

    Returns
    -------
    dict
        The extracted article date, date completed and date revised
    """
    # Get the article date and date completed
    if 'ARTICLE' not in doc['_source'] or 'ARTICLE_DATE' not in doc['_source']['ARTICLE']:
        article_date = None
    else:
        article_date = doc['_source']['ARTICLE']['ARTICLE_DATE']

    if 'DATE_COMPLETED' not in doc['_source']:
        date_completed = None
    else:
        date_completed = doc['_source']['DATE_COMPLETED']

    if 'DATE_REVISED' not in doc['_source']:
        date_revised = None
    else:
        date_revised = doc['_source']['DATE_REVISED']

    # Parse the dates if necessary
    if parse_dates:
        if article_date is not None:
            article_date = pd.to_datetime(article_date)
        if date_completed is not None:
            date_completed = pd.to_datetime(date_completed)
        if date_revised is not None:
            date_revised = pd.to_datetime(date_revised)

    return {'article_date': article_date, 'date_completed': date_completed, 'date_revised': date_revised}


def reconstruct_document_date(doc: dict, index: str = 'pubmed', time_to_complete_offset: int | None = None) -> pd.Timestamp:
    """
    Reconstruct the document date from the document

    Parameters
    ----------
    doc : dict
        The elasticsearch document
    index : str, optional
        The index to search, by default 'pubmed'
    time_to_complete_offset : int | None, optional
        The time to complete offset, by default None

    Returns
    -------
    pd.Timestamp
        The reconstructed document date
    """
    dates = extract_dates_from_document(doc)

    if time_to_complete_offset is None:
        time_to_complete_offset = get_time_to_complete_offset(index)

    if dates['article_date'] is not None:
        return pd.to_datetime(dates['article_date'])

    if dates['date_completed'] is not None:
        return pd.to_datetime(dates['date_completed']) - pd.Timedelta(days=time_to_complete_offset)

    return None


def construct_predictions_file_name(source_index: str) -> str:
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
    return f'{source_index}_predictions.json'


def construct_predictions_index_name(source_index: str, model_name: str) -> str:
    """
    Construct the name of the predictions index for the given source index

    Parameters
    ----------
    source_index: str
        The name of the source index
    model_name: str
        The name of the model

    Returns
    -------
    str
        The name of the predictions index
    """
    return f'{source_index}_{model_name}_predictions'


def check_name(name: str) -> bool:
    """
    Check whether the model name is valid (i.e. compatible with Elasticsearch index names)

    Parameters
    ----------
    name : str
        The name of the model

    Returns
    -------
    bool
        Whether the name is valid
    """
    if name is None:
        return False

    # From https://www.elastic.co/guide/en/elasticsearch/reference/7.17/indices-create-index.html
    # Index names must meet the following criteria:

    # Lowercase only
    if name.lower() != name:
        logger.error(f'Index name {name} is not lowercase')
        return False

    # Cannot include \, /, *, ?, ", <, >, |, ` ` (space character), ,, #
    for char in ['\\', '/', '*', '?', '"', '<', '>', '|', ' ', ',', '#']:
        if char in name:
            logger.error(f'Index name {name} contains invalid character {char}. Invalid characters are: \\, /, *, ?, ", <, >, |, ` ` (space character), ,, #')
            raise ValueError(f'Index name {name} contains invalid character {char}. Invalid characters are: \\, /, *, ?, ", <, >, |, ` ` (space character), ,, #')

    # Indices prior to 7.0 could contain a colon (:), but that’s been deprecated and won’t be supported in 7.0+
    if ':' in name:
        logger.error(f'Index name {name} contains invalid character :. Indices prior to 7.0 could contain a colon (:), but that\'s been deprecated and won\'t be supported in 7.0+')
        raise ValueError(f'Index name {name} contains invalid character :. Indices prior to 7.0 could contain a colon (:), but that\'s been deprecated and won\'t be supported in 7.0+')

    # Cannot start with -, _, +
    if name.startswith('-') or name.startswith('_') or name.startswith('+'):
        logger.error(f'Index name {name} starts with an invalid character. Index names cannot start with -, _, +')
        raise ValueError(f'Index name {name} starts with an invalid character. Index names cannot start with -, _, +')

    # Cannot be . or ..
    if name == '.' or name == '..':
        logger.error(f'Index name {name} is invalid. Index names cannot be . or ..')
        raise ValueError(f'Index name {name} is invalid. Index names cannot be . or ..')

    # Cannot be longer than 255 bytes (note it is bytes, so multi-byte characters will count towards the 255 limit faster)
    if len(name) > 255:
        logger.error(f'Index name {name} is too long. Index names cannot be longer than 255 bytes')
        raise ValueError(f'Index name {name} is too long. Index names cannot be longer than 255 bytes')

    # Names starting with . are deprecated, except for hidden indices and internal indices managed by plugins
    if name.startswith('.'):
        logger.error(f'Index name {name} starts with a deprecated character. Names starting with . are deprecated, except for hidden indices and internal indices managed by plugins')
        raise ValueError(f'Index name {name} starts with a deprecated character. Names starting with . are deprecated, except for hidden indices and internal indices managed by plugins')

    return True


def download_model(author: str, model: str, target_dir: str, overwrite: bool = False) -> None:
    """
    Download a model from the Huggingface Hub

    Parameters
    ----------
    author : str
        The author of the model
    model : str
        The name of the model
    target_dir : str | None
        The local directory to download the model to
    overwrite : bool, optional
        Whether to overwrite the local directory if it already exists, by default False
    """
    if os.path.exists(target_dir) and os.listdir(target_dir):
        if overwrite:
            logger.info(f'Model {model} already exists. Overwriting.')
            shutil.rmtree(target_dir)
        else:
            logger.info(f'Model {model} already exists. Skipping.')
            return

    BASE_URL = "https://huggingface.co"
    REPO_ID = f"{author}/{model}"
    MODEL_URL = f"{BASE_URL}/{REPO_ID}"

    # Download the model
    os.system(f'git clone {MODEL_URL} {target_dir}')

    # Remove the .git directory
    shutil.rmtree(os.path.join(target_dir, '.git'))

    # Remove the .gitattributes and README.md files
    os.remove(os.path.join(target_dir, '.gitattributes'))
    os.remove(os.path.join(target_dir, 'README.md'))
