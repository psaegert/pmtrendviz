import logging
from typing import Any

from ..models.factory import ModelFactory
from ..models.manager import ModelManager
from ..utils import get_es_client

logger = logging.getLogger('pmtrendviz.app.trend')


class TrendGenerator:
    def __init__(self, source_index: str = 'pubmed', resolution: str = 'M', n_closest_clusters: int = 4, ignore_top_n_dates: int = 0, distance: str = 'cosine'):
        self.model: ModelManager | None = None
        self.model_name: str | None = None
        self.source_index = source_index

        self.query = ''
        self.resolution = resolution
        self.distance = distance
        self.n_closest_clusters = n_closest_clusters
        self.ignore_top_n_dates = ignore_top_n_dates

    def load_model(self, model_name: str) -> None:
        """
        Load the model

        Parameters
        ----------
        model_name : str
            The name of the model to load, by default None

        Returns
        -------
        None
        """
        self.model_name = model_name
        self.model = ModelFactory.load(model_name)
        self.predictions_index = self.model._construct_predictions_index_name(self.source_index)

        logger.debug(f'TrendGenerator loaded model {model_name}')

    def generate_figure(self) -> list[dict[str, Any]]:
        """
        Generate the figure for the given query

        Returns
        -------
        dict
            The figure
        """
        if self.model is None or not self.query:
            return []

        # Check if the model has precomputed predictions
        if self.model is None:
            logger.error('No model loaded')
            raise ValueError('No model loaded')

        es = get_es_client()

        # Check if the predictions index exists
        if not es.indices.exists(index=self.predictions_index):
            logger.error(f'Predictions index {self.predictions_index} does not exist')
            raise ValueError(f'Predictions index {self.predictions_index} does not exist')

        # Get the trends for the given query
        trends = self.model.generate_trends(self.query, self.resolution, self.source_index, self.distance, self.n_closest_clusters, self.ignore_top_n_dates, es)

        # Generate the figure
        return [{
                'x': trends['date'].to_list(),
                'y': trends[column].tolist(),
                'type': 'line',
                'name': self.model.cluster_names.loc[int(column.split('_')[1])]['name']
                } for column in trends.columns if column != 'date'
                ]
