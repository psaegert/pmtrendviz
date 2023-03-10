from typing import Literal

from pydantic import BaseModel, Field


class GeneralArgs(BaseModel):
    index: str = Field(description='The ElasticSearch index to use')
    random_state: int = Field(default=42, description='The random state to use for the pipeline')
    sample_method: list[Literal['uniform', 'forward', 'backward']] = Field(default='uniform', description='The method to use for sampling the data.')
    n_samples: int = Field(default=1e6, description='The number of samples to use for training')
    model_save_path: str = Field(description='The path to save the model to')
    overwrite: bool = Field(default=False, description='Whether to overwrite the model if it already exists')
