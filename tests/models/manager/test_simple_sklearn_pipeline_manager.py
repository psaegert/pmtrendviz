import os
import shutil
import time
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from pmtrendviz.models.factory import ModelFactory
from pmtrendviz.models.manager import SimpleSklearnPipelineManager
from pmtrendviz.train.data import sample_training_data
from pmtrendviz.utils import get_es_client, get_models_dir


@pytest.mark.usefixtures('manage_pytest_index', 'manage_prediction_indices')
class TestSimpleSklearnPipelineManager:
    sample_data: pd.DataFrame = pd.DataFrame()
    n_samples: int = 1000
    index: str = 'pytest'
    model_name: str = 'pytest-model'
    model = 'tfidf_truncatedsvd_kmeans'
    stop_words = 'english'
    max_df: float = 0.999
    min_df: float = 0.001
    n_components: int = 100
    n_clusters: int = 69
    ngram_range: Tuple[int, int] = (1, 1)
    random_state: int = 42
    include_title: bool = True
    include_abstract: bool = True
    include_keywords_major: bool = False
    include_keywords_minor: bool = False

    @classmethod
    def setup_class(cls) -> None:
        # Clear the test model directory
        shutil.rmtree(os.path.join(get_models_dir(), cls.model_name), ignore_errors=True)

        # Sample some data
        cls.sample_data = sample_training_data(cls.index, random_state=42, n_samples=cls.n_samples, method='forward')['text']

    @classmethod
    def teardown_class(cls) -> None:
        # Clear the test model directory
        shutil.rmtree(os.path.join(get_models_dir(), cls.model_name), ignore_errors=True)

    def teardown_method(self) -> None:
        # Clear the test model directory
        shutil.rmtree(os.path.join(get_models_dir(), self.model_name), ignore_errors=True)

    def test_sample_data(self) -> None:
        assert len(self.sample_data) == self.n_samples

    def test_create_default_simple_sklearn_pipeline_manager(self) -> None:
        manager = ModelFactory.create(
            self.model,
            self.include_title,
            self.include_abstract,
            self.include_keywords_major,
            self.include_keywords_minor,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            n_components=self.n_components,
            n_clusters=self.n_clusters,
            ngram_range=self.ngram_range,
            random_state=None)

        assert manager is not None
        assert isinstance(manager.model, Pipeline)
        assert manager.random_state is None
        assert manager.preprocessing_args == {
            'include_title': True,
            'include_abstract': True,
            'include_keywords_major': False,
            'include_keywords_minor': False,
        }

        assert (manager.cluster_names.columns == ['name']).all()
        assert len(manager.cluster_names) == 0

    def test_create_custom_simple_sklearn_pipeline_manager(self) -> None:
        manager = ModelFactory.create(
            self.model,
            self.include_title,
            self.include_abstract,
            True,
            self.include_keywords_minor,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            n_components=self.n_components,
            n_clusters=self.n_clusters,
            ngram_range=self.ngram_range,
            random_state=42)

        assert manager.random_state == 42
        assert manager.preprocessing_args == {
            'include_title': True,
            'include_abstract': True,
            'include_keywords_major': True,
            'include_keywords_minor': False,
        }

    def test_set_name(self) -> None:
        manager = ModelFactory.create(
            self.model,
            self.include_title,
            self.include_abstract,
            self.include_keywords_major,
            self.include_keywords_minor,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            n_components=self.n_components,
            n_clusters=self.n_clusters,
            ngram_range=self.ngram_range,
            random_state=self.random_state)
        manager.set_name(self.model_name)

        assert manager._name == self.model_name

    def test_construct_predictions_index_name(self) -> None:
        manager = ModelFactory.create(
            self.model,
            self.include_title,
            self.include_abstract,
            self.include_keywords_major,
            self.include_keywords_minor,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            n_components=self.n_components,
            n_clusters=self.n_clusters,
            ngram_range=self.ngram_range,
            random_state=self.random_state)
        manager.set_name(self.model_name)
        predictions_index_name = manager._construct_predictions_index_name(self.index)
        predictions_file_name = manager._construct_predictions_file_name(self.index)

        assert manager._name == self.model_name
        assert predictions_index_name == f'{self.index}_{manager._name}_predictions'
        assert predictions_file_name == f'{self.index}_predictions.json'

    def test_fit(self) -> None:
        manager = ModelFactory.create(
            self.model,
            self.include_title,
            self.include_abstract,
            self.include_keywords_major,
            self.include_keywords_minor,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            n_components=self.n_components,
            n_clusters=self.n_clusters,
            ngram_range=self.ngram_range,
            random_state=self.random_state)

        manager.fit(self.sample_data)

        assert manager.model['clustering_kmeans'].labels_.shape == (self.n_samples,)
        assert len(manager.cluster_names) == self.n_clusters

    def test_predict(self) -> None:
        manager = ModelFactory.create(
            self.model,
            self.include_title,
            self.include_abstract,
            self.include_keywords_major,
            self.include_keywords_minor,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            n_components=self.n_components,
            n_clusters=self.n_clusters,
            ngram_range=self.ngram_range,
            random_state=self.random_state)

        manager.fit(self.sample_data)
        predictions = manager.predict(self.sample_data)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (self.n_samples,)

    def test_transform(self) -> None:
        manager = ModelFactory.create(
            self.model,
            self.include_title,
            self.include_abstract,
            self.include_keywords_major,
            self.include_keywords_minor,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            n_components=self.n_components,
            n_clusters=self.n_clusters,
            ngram_range=self.ngram_range,
            random_state=self.random_state)

        manager.fit(self.sample_data)
        transformed = manager.transform(self.sample_data, exclude_from='clustering')

        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == (self.n_samples, self.n_components)

        transformed_with_kmeans = manager.transform(self.sample_data)

        assert isinstance(transformed_with_kmeans, np.ndarray)
        assert transformed_with_kmeans.shape == (self.n_samples, self.n_clusters)

    def test_save(self) -> None:
        manager = ModelFactory.create(
            self.model,
            self.include_title,
            self.include_abstract,
            self.include_keywords_major,
            self.include_keywords_minor,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            n_components=self.n_components,
            n_clusters=self.n_clusters,
            ngram_range=self.ngram_range,
            random_state=self.random_state)

        manager.fit(self.sample_data)
        manager.save(self.model_name)

        assert os.path.exists(os.path.join(get_models_dir(), self.model_name, 'model.joblib'))
        assert os.path.exists(os.path.join(get_models_dir(), self.model_name, 'config.json'))
        assert os.path.exists(os.path.join(get_models_dir(), self.model_name, 'cluster_names.csv'))

    def test_save_only_model(self) -> None:
        manager = ModelFactory.create(
            self.model,
            self.include_title,
            self.include_abstract,
            self.include_keywords_major,
            self.include_keywords_minor,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            n_components=self.n_components,
            n_clusters=self.n_clusters,
            ngram_range=self.ngram_range,
            random_state=self.random_state)

        manager.fit(self.sample_data)
        manager.save(self.model_name, save_only='model')

        assert os.path.exists(os.path.join(get_models_dir(), self.model_name, 'model.joblib'))
        assert not os.path.exists(os.path.join(get_models_dir(), self.model_name, 'config.json'))
        assert not os.path.exists(os.path.join(get_models_dir(), self.model_name, 'cluster_names.csv'))

    def test_load(self) -> None:
        manager = ModelFactory.create(
            self.model,
            self.include_title,
            self.include_abstract,
            True,
            self.include_keywords_minor,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            n_components=self.n_components,
            n_clusters=self.n_clusters,
            ngram_range=self.ngram_range,
            random_state=self.random_state)

        manager.fit(self.sample_data)
        manager.save(self.model_name)

        manager_predictions = manager.model.predict(self.sample_data)
        manager_transformed = manager.model.transform(self.sample_data)

        loaded_manager = SimpleSklearnPipelineManager.load(self.model_name)

        loaded_manager_predictions = loaded_manager.model.predict(self.sample_data)
        loaded_manager_transformed = loaded_manager.model.transform(self.sample_data)

        assert isinstance(loaded_manager, SimpleSklearnPipelineManager)
        assert isinstance(loaded_manager.model, Pipeline)
        assert loaded_manager.random_state == 42
        assert loaded_manager.preprocessing_args == {
            'include_title': True,
            'include_abstract': True,
            'include_keywords_major': True,
            'include_keywords_minor': False,
        }

        assert (loaded_manager.cluster_names.columns == ['name']).all()
        assert len(loaded_manager.cluster_names) == self.n_clusters

        assert (loaded_manager_predictions == manager_predictions).all()
        assert (loaded_manager_transformed == manager_transformed).all()

    def test_load_from_factory(self) -> None:
        manager = ModelFactory.create(
            self.model,
            self.include_title,
            self.include_abstract,
            True,
            self.include_keywords_minor,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            n_components=self.n_components,
            n_clusters=self.n_clusters,
            ngram_range=self.ngram_range,
            random_state=self.random_state)

        manager.fit(self.sample_data)
        manager.save(self.model_name)

        loaded_manager = ModelFactory.load(self.model_name)

        assert isinstance(loaded_manager, SimpleSklearnPipelineManager)
        assert isinstance(loaded_manager.model, Pipeline)
        assert loaded_manager.random_state == 42
        assert loaded_manager.preprocessing_args == {
            'include_title': True,
            'include_abstract': True,
            'include_keywords_major': True,
            'include_keywords_minor': False,
        }

        assert (loaded_manager.cluster_names.columns == ['name']).all()
        assert len(loaded_manager.cluster_names) == self.n_clusters

    def test_evalutate(self) -> None:
        manager = ModelFactory.create(
            self.model,
            self.include_title,
            self.include_abstract,
            self.include_keywords_major,
            self.include_keywords_minor,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            n_components=self.n_components,
            n_clusters=self.n_clusters,
            ngram_range=self.ngram_range,
            random_state=self.random_state)

        manager.fit(self.sample_data)
        metrics = manager.evaluate(self.sample_data)

        assert isinstance(metrics, pd.DataFrame)
        assert 'Silhouette' in metrics.columns
        assert 'Calinski-Harabasz' in metrics.columns
        assert 'Davies-Bouldin' in metrics.columns
        assert len(metrics) == 1

    def test_precompute_predictions_max_new_predictions_amount_limited(self) -> None:
        manager = ModelFactory.create(
            self.model,
            self.include_title,
            self.include_abstract,
            self.include_keywords_major,
            self.include_keywords_minor,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            n_components=self.n_components,
            n_clusters=self.n_clusters,
            ngram_range=self.ngram_range,
            random_state=self.random_state)

        es = get_es_client()

        manager.fit(self.sample_data)
        manager.set_name(self.model_name)
        manager.precompute_predictions(source_index=self.index, max_new_predictions=1_000, batch_size=100, es=es)
        predictions_index_name = manager._construct_predictions_index_name(self.index)

        # Refresh the index
        es.indices.refresh(index=predictions_index_name)

        assert es.indices.exists(index=predictions_index_name)
        assert es.count(index=predictions_index_name)['count'] == 1_000

        # Check if the mapping is correct
        mapping = es.indices.get_mapping(index=predictions_index_name)[predictions_index_name]['mappings']['properties']
        assert mapping['date']['type'] == 'date'
        assert mapping['label']['type'] == 'integer'
        assert mapping['PMID']['type'] == 'keyword'

        manager.clear_predictions(self.index, es)

        assert not es.indices.exists(index=predictions_index_name)

    def test_precompute_predictions_max_predictions_time_limited(self) -> None:
        manager = ModelFactory.create(
            self.model,
            self.include_title,
            self.include_abstract,
            self.include_keywords_major,
            self.include_keywords_minor,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            n_components=self.n_components,
            n_clusters=self.n_clusters,
            ngram_range=self.ngram_range,
            random_state=self.random_state)

        es = get_es_client()

        manager.fit(self.sample_data)
        manager.set_name(self.model_name)

        start_time = time.time()
        manager.precompute_predictions(self.index, timeout=5, batch_size=100, es=es)
        end_time = time.time()

        predictions_index_name = manager._construct_predictions_index_name(self.index)

        # Refresh the index
        es.indices.refresh(index=predictions_index_name)

        assert end_time - start_time < 5 + 30  # 30 seconds tolerance for slow machines

        assert es.indices.exists(index=predictions_index_name)
        assert es.count(index=predictions_index_name)['count'] > 0

        # Check if the mapping is correct
        mapping = es.indices.get_mapping(index=predictions_index_name)[predictions_index_name]['mappings']['properties']
        assert mapping['date']['type'] == 'date'
        assert mapping['label']['type'] == 'integer'
        assert mapping['PMID']['type'] == 'keyword'

        manager.clear_predictions(self.index, es)

        assert not es.indices.exists(index=predictions_index_name)

    def test_get_label_trend(self) -> None:
        manager = ModelFactory.create(
            self.model,
            self.include_title,
            self.include_abstract,
            self.include_keywords_major,
            self.include_keywords_minor,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            n_components=self.n_components,
            n_clusters=self.n_clusters,
            ngram_range=self.ngram_range,
            random_state=self.random_state)
        es = get_es_client()

        manager.fit(self.sample_data)
        manager.set_name(self.model_name)

        manager.precompute_predictions(self.index, max_new_predictions=1_000, es=es)
        predictions_index_name = manager._construct_predictions_index_name(self.index)

        # Refresh the index
        es.indices.refresh(index=predictions_index_name)

        all_trends_empty = True
        for label in range(self.n_clusters):
            if len(manager._get_label_trend(self.index, label, 'year', ignore_top_n_dates=0, es=es)) > 0:
                all_trends_empty = False
                break

        assert not all_trends_empty
        assert isinstance(manager._get_label_trend(self.index, 0, 'year', ignore_top_n_dates=0, es=es), pd.DataFrame)

        manager.clear_predictions(self.index, es)

        assert not es.indices.exists(index=predictions_index_name)

    def test_export_import_predictions(self) -> None:
        manager = ModelFactory.create(
            self.model,
            self.include_title,
            self.include_abstract,
            self.include_keywords_major,
            self.include_keywords_minor,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            n_components=self.n_components,
            n_clusters=self.n_clusters,
            ngram_range=self.ngram_range,
            random_state=self.random_state)
        es = get_es_client()

        manager.fit(self.sample_data)
        manager.set_name(self.model_name)

        manager.precompute_predictions(self.index, max_new_predictions=1_000, es=es)
        predictions_index_name = manager._construct_predictions_index_name(self.index)

        # Refresh the index
        es.indices.refresh(index=predictions_index_name)

        manager.export_predictions(self.index, es)

        filepath = os.path.join(get_models_dir(), manager._name, manager._construct_predictions_file_name(self.index))
        assert os.path.exists(filepath)

        manager.clear_predictions(self.index, es)

        assert not es.indices.exists(index=predictions_index_name)

        manager.import_predictions(filepath=filepath, es=es)

        # Refresh the index
        es.indices.refresh(index=predictions_index_name)

        assert es.indices.exists(index=predictions_index_name)
        assert es.count(index=predictions_index_name)['count'] == 1_000

        manager.clear_predictions(self.index, es)

        assert not es.indices.exists(index=predictions_index_name)

        manager.import_predictions(source_index=self.index, es=es)

        # Refresh the index
        es.indices.refresh(index=predictions_index_name)

        assert es.indices.exists(index=predictions_index_name)
        assert es.count(index=predictions_index_name)['count'] == 1_000

        manager.clear_predictions(self.index, es)

        assert not es.indices.exists(index=predictions_index_name)

        os.remove(filepath)
