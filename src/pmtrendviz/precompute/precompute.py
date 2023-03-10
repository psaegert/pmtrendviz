import argparse
import logging

from ..models.factory import ModelFactory
from ..utils import get_es_client

logger = logging.getLogger('pmtrendviz.precompute.precompute')


def precompute(args: argparse.Namespace) -> None:
    """
    Precompute the predictions

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments
    """

    # Load the model
    logger.debug('Loading the model')
    manager = ModelFactory.load(args.model_name)

    # Get the Elasticsearch client
    logger.debug('Getting the Elasticsearch client')
    es = get_es_client()

    manager.precompute_predictions(args.index, args.max_new_predictions, args.timeout, args.batch_size, args.sample_method, es=es)
