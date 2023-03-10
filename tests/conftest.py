import os
import shutil
from argparse import Namespace
from typing import Generator

import pytest
import spacy
from elasticsearch.exceptions import NotFoundError

from pmtrendviz.import_pipeline.end_to_end import import_end_to_end
from pmtrendviz.utils import get_data_dir, get_es_client


def manage_index(index: str) -> Generator:
    """
    Create an index for the tests to use

    Parameters
    ----------
    index : str, optional
        The name of the index to create, by default 'pytest'

    Yields
    ------
    Generator
        The index name
    """
    es = get_es_client()

    # Try to query the index to see if it exists
    try:
        es.count(index=index)
    except NotFoundError:
        # If the index doesn't exist, create it
        namespace = Namespace(
            last_n_files=1,
            yes=True,
            index=index,
            n_threads=1,
            max_retries=3,
            backoff_factor=300,
        )

        import_end_to_end(namespace)

    # Wait for the index to be ready
    es.indices.refresh(index=index)

    yield index

    # Delete the index after the tests are done
    es.indices.delete(index=index)

    # Delete the file that lists the imported files (not necessary, but for cleanliness)
    if os.path.exists(os.path.join(get_data_dir(), 'imported_files', f'{index}.txt')):
        os.remove(os.path.join(get_data_dir(), 'imported_files', f'{index}.txt'))


@pytest.fixture(scope='session')
def manage_pytest_index() -> Generator:
    """
    Create an index for the tests to use

    Yields
    ------
    Generator
        The index name
    """
    yield from manage_index('pytest')


@pytest.fixture(scope='session')
def manage_prediction_indices() -> Generator:
    """
    Delete the prediction indices after the tests are done

    Yields
    ------
    Generator
        None
    """

    yield

    # Delete the prediction indices
    es = get_es_client()

    # Get all test prediction indices (e.g. 'pytest_pytest-model_predictions')
    indices = es.indices.get_alias(index='*pytest-model_predictions')

    for index in indices:
        es.indices.delete(index=index, ignore_unavailable=True)


@pytest.fixture(scope='session')
def install_en_core_sci_sm() -> None:
    """
    Install the en-core-sci-sm model
    """
    # Install the spacy model
    # pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
    try:
        spacy.load('en_core_sci_sm')
    except OSError:
        os.system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz')


@pytest.fixture(scope='session')
def manage_pytest_folder() -> Generator:
    """
    Create a folder for the tests to use

    Yields
    ------
    Generator
        The folder name
    """
    folder = 'pytest'

    # Create the folder
    os.makedirs(folder, exist_ok=True)

    yield folder

    # Delete the folder after the tests are done
    if os.path.exists(folder):
        shutil.rmtree(folder)
