import inspect
import json
import os
from collections import defaultdict
from logging import getLogger

from fastapi import APIRouter

from pmtrendviz.models.factory import ModelFactory
from pmtrendviz.utils import get_models_dir

logger = getLogger('pmtrendviz.api.routers.models')


def fetch_available_models():  # type: ignore
    models_dir = get_models_dir()
    models = []
    for model in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model)
        if os.path.isdir(model_path):
            with open(os.path.join(model_path, 'config.json')) as config_fp:
                models.append({
                    'name': model,
                    'path': model_path,
                    'config': json.load(config_fp)
                })

    logger.debug(f'Found {len(models)} models: {models}')
    return models


router = APIRouter(
    prefix="/models",
    tags=["models"]
)


@router.get("/public")
def get_public_models():  # type: ignore
    raise NotImplementedError


@router.get("/saved")
def get_saved_models():  # type: ignore
    models = fetch_available_models()
    logger.debug(models)
    return models


@router.get("/untrained")
def get_untrained_models():  # type: ignore
    models = defaultdict(list)
    for name, _f in inspect.getmembers(ModelFactory, inspect.isfunction):
        if name not in ['create', 'load', 'remove', 'install']:
            fullargspec = inspect.getfullargspec(_f)
            for arg_name, arg_type in fullargspec.annotations.items():
                if arg_name == 'return':
                    continue
                else:
                    models[name].append(arg_type.schema())
    return models
