from abc import ABC
from typing import Literal

from pydantic import BaseModel, Field


class ModelBaseArgs(ABC, BaseModel):
    stop_words: str = Field(default='english', description='The stop words to use')
    min_df: float = Field(default=0.001, description='The minimum document frequency')
    max_df: float = Field(default=0.999, description='The maximum document frequency')
    ngram_range: tuple[int, int] = Field(default=(1, 1), description='The n-gram range to use for the TF-IDF')
    n_components: int = Field(default=100, description='The number of components to use for the SVD')
    n_clusters: int = Field(default=100, description='The number of clusters to use for the KMeans')


class TfidfTruncatedSvdKmeansArgsModel(ModelBaseArgs):
    class Config:
        title = 'TFIDF-TruncatedSVD-KMeans'


class NerTfidfTruncatedSvdKmeansArgsModel(ModelBaseArgs):
    spacy_model: str = Field(default='en_core_sci_sm', description='The spacy model to use for preprocessing')
    spacy_disable: list[Literal["tokenizer", "tagger", "parser", "ner", "lemmatizer", "textcat", "custom"]] = Field(default=[], description='The spacy components to disable')

    class Config:
        title = 'NER-TFIDIF-TruncatedSVD-KMeans'


class PreProcessingArgs(BaseModel):
    include_title: bool = Field(default=True, description='Whether to include the title in the training data')
    include_abstract: bool = Field(default=True, description='Whether to include the abstract in the training data')
    include_keywords_major: bool = Field(default=False, description='Whether to include the major keywords in the training data')
    include_keywords_minor: bool = Field(default=False, description='Whether to include the minor keywords in the training data')

    class Config:
        title = 'Preprocessing'
