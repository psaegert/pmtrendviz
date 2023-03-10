import argparse
import json
import logging
import os
from typing import Dict, Iterable

from elasticsearch.helpers import parallel_bulk
from tqdm import tqdm

from elasticsearch import Elasticsearch

from ..utils import UserAbort, get_es_client

logger = logging.getLogger('pmtrendviz.import_pipeline.index')
# filter out log records from elasticsearch
logging.getLogger('elasticsearch').setLevel(logging.WARNING)
# filter out log records from urllib3
logging.getLogger('urllib3').setLevel(logging.WARNING)


def generate_documents(filepath: str, skip_odd: bool = True) -> Iterable[Dict]:
    """
    Creates a generator that yields the articles from the file as a dictionary

    Parameters
    ----------
    filepath: str
        The path to the file to import
    Returns
    -------
    Iterable[Dict]
        A generator that yields the articles from the file as a dictionary
    """
    with open(filepath, 'r') as f:
        for i, doc_line in enumerate(f, start=int(skip_odd)):
            if i % 2 == 0 or not skip_odd:
                doc = json.loads(doc_line)
                doc['_id'] = doc['PMID']
                yield doc


def import_file(filepath: str, client: Elasticsearch, index: str, skip_odd: bool = True) -> None:
    """
    Imports a file into the elasticsearch index

    Parameters
    ----------
    index:str
        The name of the index to import the data to
    client: Elasticsearch
        The client wrapper to use for the import
    filepath: str
        The path to the file to import
    """
    # Use bulk helper to import the data
    returns = parallel_bulk(client, generate_documents(filepath, skip_odd), index=index)
    error = False
    for ok, result in returns:
        if not ok:
            logger.error(result)
            error = True
    if not error:
        logger.info(f'Imported {filepath} successfully')


def create_index(client: Elasticsearch, index: str = 'pubmed') -> None:
    """
    Creates the index for the PubMed data
    Parameters
    ----------
    client: Elasticsearch
        The client wrapper to use for the import

    """
    logger.debug(f'Creating index {index}')
    # get return value of the create index request

    response = client.indices.create(
        index=index,
        settings={
            'index': {
                'number_of_shards': 1,
                'number_of_replicas': 0,
            },
        },
        ignore=400, )  # ignore error if index already exists
    # analyze response of the create index request
    if 'acknowledged' in response:
        logger.debug(f'Index {index} created')
    else:
        logger.debug(f'Index {index} already exists')


def index(args: argparse.Namespace) -> None:
    """
    Import the extracted data into an Elasticsearch index

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments
    """
    # Get the client
    es = get_es_client()

    # Collect the file paths
    filepaths = [os.path.join(args.src_dir, f) for f in sorted(os.listdir(args.src_dir)) if f.endswith('.json')]

    # Only import the last n files
    if args.last_n_files is not None:
        filepaths = filepaths[-args.last_n_files:]

    # Ask the user if they want to continue
    if not args.yes:
        answer = input(f'Indexing data from {len(filepaths)} files into the {args.index} index. Continue? (y/n) ')
        if answer.lower() not in ['y', 'yes']:
            raise UserAbort("Confirmation of index")

    # Create the index
    create_index(es, index=args.index)

    # Import the files
    for filename in tqdm(filepaths):
        import_file(filename, client=es, index=args.index)
