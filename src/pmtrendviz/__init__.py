# create, load, download, remove models

from typing import Any

from .models.factory import ModelFactory
from .models.manager import ModelManager

__version__ = "0.1.0"


def create(
    model: str,
    include_title: bool = True,
    include_abstract: bool = True,
    include_keywords_major: bool = False,
    include_keywords_minor: bool = False,
    **kwargs: Any
) -> ModelManager:
    """
    Create a new model from the given name using the ModelFactory

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
        The model manager for the created model
    """
    return ModelFactory.create(model, include_title, include_abstract, include_keywords_major, include_keywords_minor, **kwargs)


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
    return ModelFactory.load(name)


def remove(name: str, ignore_errors: bool = False) -> None:
    """
    Remove a model from the models directory and delete all prediction indices associated with it

    Parameters
    ----------
    name : str
        The name of the model to remove
    ignore_errors : bool
        Whether to ignore errors when removing the model
    """
    ModelFactory.remove(name, ignore_errors=ignore_errors)


def install(name: str) -> None:
    """
    Install a model from the models directory

    Parameters
    ----------
    name : str
        The name of the model to install
    """
    ModelFactory.install(name)
