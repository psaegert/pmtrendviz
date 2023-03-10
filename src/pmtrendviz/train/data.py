import logging
import os
from typing import Mapping

import numpy as np
import pandas as pd
from tqdm import tqdm

from elasticsearch import helpers

from ..utils import get_cache_dir, get_es_client
from .preprocessing import combine_text

logger = logging.getLogger('pmtrendviz.train.data')


def sample_training_data(index: str, random_state: int | None = None, combine_kwargs: Mapping[str, bool] | None = None, n_samples: int = 1_000_000, method: str = 'uniform', cache_file: str | None = None) -> pd.DataFrame:
    """
    Get the training data

    Parameters
    ----------
    index : str
        The index to sample from
    random_state | None : int
        The random state to use. Only affects 'uniform' sampling
    combine_kwargs : Dict[str, bool] | None, optional
        The keyword arguments to pass to combine_text
    n_samples : int, optional
        The number of samples to take, by default 1_000_000
    method : {'uniform', 'forward', 'backward'}, optional
        The method to use for sampling, by default 'uniform'. 'forward' is fastest because it does not scan the entire index.
    cache : str | None , optional
        The name of the cache file to use, by default None

    Returns
    -------
    pd.DataFrame
        The training data
    """
    if cache_file is not None:
        cache_dir = get_cache_dir()
        cache_path = os.path.join(cache_dir, 'data', cache_file)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # Load from cache if it exists
        if os.path.exists(cache_path):
            logger.info(f'Loading training data from cache at {cache_path}')
            return pd.read_csv(cache_path, index_col=0)

    es = get_es_client()

    # Total count up to which to sample indices/ids
    result = es.cat.count(index=index, params={"format": "json"})
    total_document_count = int(result[0]['count'])  # type: ignore

    logger.info(f"Sampling {n_samples} from {total_document_count} total documents")

    # Randomly sample N documents (max integer: count) and sort them
    if random_state is not None:
        np.random.seed(int(random_state))
        logger.debug(f"Using random state {random_state}")
    else:
        logger.debug('Using random state None (randomly generate)')

    match method:
        case 'uniform':
            training_indices = np.sort(np.random.choice(total_document_count, int(n_samples), replace=False))
        case 'forward':
            training_indices = np.arange(n_samples)
            total_document_count = n_samples
        case 'backward':
            training_indices = np.arange(total_document_count - n_samples, total_document_count)  # HACK: Find a way to speed this option up
        case _:
            logger.error(f"Invalid method: {method}")
            raise ValueError(f"Invalid method: {method}")

    pbar = tqdm(
        helpers.scan(
            es,
            index=index,
            query={'query': {'match_all': {}}}, scroll='1m'),
        total=total_document_count,
        desc=f'Sampling training data in {method} mode')

    training_data_dict = {}

    j = 0  # Increments when a new training document is found
    for i, doc in enumerate(pbar):
        if i == training_indices[j]:
            training_data_dict[i] = {
                'PMID': doc['_source']['PMID'],
                'text': combine_text(doc) if combine_kwargs is None else combine_text(doc, **combine_kwargs),
            }

            j += 1

            if j == len(training_indices):
                break
    pbar.close()

    df = pd.DataFrame.from_dict(training_data_dict, orient='index')

    if cache_file is not None:
        logger.info(f"Caching training data to {cache_file}")
        df.to_csv(cache_path)

    return df
