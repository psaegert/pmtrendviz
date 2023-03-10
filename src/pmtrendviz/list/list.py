import argparse
import logging
import os

from ..utils import construct_predictions_index_name, get_es_client

logger = logging.getLogger('pmtrendviz.list')


def list_managers() -> None:
    """
    List the implemented managers
    """
    from ..models import manager as mgr

    # Get all classes defined in the module
    classes = [getattr(mgr, name) for name in dir(mgr) if isinstance(getattr(mgr, name), type)]

    # Filter the classes to only get the managers
    managers = [cls for cls in classes if issubclass(cls, mgr.ModelManager) and not cls == mgr.ModelManager]

    # Collect the names of the managers
    managers_list = [manager.__name__ for manager in managers]

    # Print the managers
    print(f'Implemented managers ({len(managers_list)}):')
    for manager in managers_list:
        print('-', manager)
    print()


def list_trainable_models() -> None:
    """
    List the trainable models
    """
    from ..models.factory import ModelFactory

    def filter_trainable_models(method_name: str) -> bool:
        if not callable(getattr(ModelFactory, method_name)) or method_name.startswith('_') or method_name in ['load', 'create', 'install', 'remove']:
            return False

        return True

    # Get the factory methods
    factory_methods_list = [getattr(ModelFactory, name).__name__ for name in dir(ModelFactory) if filter_trainable_models(name)]

    # Print the factory methods
    print(f'Trainable models ({len(factory_methods_list)}):')
    for factory_method in factory_methods_list:
        print('-', factory_method)
    print()


def list_saved_models(with_predictions: bool = False) -> None:
    """
    List the saved models
    """
    from ..utils import get_models_dir

    # Get the models directory
    models_dir = get_models_dir()

    if with_predictions:
        es = get_es_client()

    # Collect the saved models
    saved_models_list = []
    for model_name in os.listdir(models_dir):
        if not os.path.isdir(os.path.join(models_dir, model_name)):
            continue

        # If the model has no predictions, skip it
        if with_predictions:
            if len(es.indices.get(index=construct_predictions_index_name(source_index='*', model_name=model_name))) == 0:
                continue

        saved_models_list.append(model_name)

    # Print the saved models
    print(f'Saved models{" with predictions" if with_predictions else ""} ({len(saved_models_list)}):')
    for model_name in saved_models_list:
        print('-', model_name)
    print()


def list_pretrained_models() -> None:
    """
    List the pretrained models
    """
    # List all the available models from psaegert for pmtrendviz
    from huggingface_hub import HfApi

    all_models_list = HfApi().list_models(author='psaegert')

    # Filter the models for pmtrendviz
    pretrained_models_list = [model for model in all_models_list if model.modelId.startswith('psaegert/pmtrendviz-')]

    # Print the models
    print(f'pretrained models ({len(pretrained_models_list)}):')
    for model_name in pretrained_models_list:
        print('-', model_name.modelId[len('psaegert/pmtrendviz-'):])
    print()


def list_(args: argparse.Namespace) -> None:
    """
    List the available models and managers

    Parameters
    ----------
    args : argparse.Namespace
        The arguments

    Returns
    -------
    None
    """
    if args.managers:
        logger.debug('Listing the implemented managers')
        list_managers()
    if args.trainable:
        logger.debug('Listing the trainable models')
        list_trainable_models()
    if args.saved:
        logger.debug('Listing the saved models')
        list_saved_models()
    if args.pretrained:
        logger.debug('Listing the pretrained models')
        list_pretrained_models()
    if args.with_predictions:
        logger.debug('Listing the models with predictions')
        list_saved_models(with_predictions=True)

    if not any([args.managers, args.trainable, args.saved, args.pretrained, args.with_predictions]):
        logger.debug('Listing all available models')
        list_managers()
        list_trainable_models()
        list_saved_models()
        list_saved_models(with_predictions=True)
        list_pretrained_models()
