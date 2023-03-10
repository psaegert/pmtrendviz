import argparse
import logging
import os

from ..models.factory import ModelFactory
from .data import sample_training_data

logger = logging.getLogger('pmtrendviz.train.train')

CPU_COUNT = os.cpu_count() or 1


def train(args: argparse.Namespace) -> None:
    # Get the preprocessing args
    preprocessing_kwargs = {
        'include_title': args.include_title,
        'include_abstract': args.include_abstract,
        'include_keywords_major': args.include_keywords_major,
        'include_keywords_minor': args.include_keywords_minor,
    }

    # Get the model manager
    logger.debug('Creating model manager')
    model_manager = ModelFactory.create(**vars(args))  # HACK: The relevant arguments should be passed explicitly before the train method

    # Sample the data
    logger.debug('Sampling the data')
    data = sample_training_data(args.index, args.random_state, preprocessing_kwargs, args.n_samples, args.sample_method)

    # Train the model
    logger.info('Training the model')
    model_manager.fit(data['text'])

    # Save the model
    logger.info('Saving the model')
    model_manager.save(args.save, overwrite=args.overwrite)
