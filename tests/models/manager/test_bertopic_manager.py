import os
import shutil

import numpy as np
import pandas as pd
import pytest
from bertopic import BERTopic

from pmtrendviz.models.factory import ModelFactory
from pmtrendviz.models.manager import BERTopicManager
from pmtrendviz.train.data import sample_training_data
from pmtrendviz.utils import get_es_client, get_models_dir


@pytest.mark.usefixtures('manage_pytest_index', 'manage_prediction_indices')
class TestBERTopicManager:
    sample_data: pd.DataFrame = pd.DataFrame()
    n_samples: int = 150
    index: str = 'pytest'
    model_name: str = 'pytest-model'
    model: str = 'default_bertopic'
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

    def test_create_default_bertopic_manager(self) -> None:
        manager = ModelFactory.create(self.model, self.include_title, self.include_abstract, self.include_keywords_major, self.include_keywords_minor)
        assert manager is not None
        assert isinstance(manager.model, BERTopic)
        assert manager.random_state is None
        assert manager.preprocessing_args == {
            'include_title': True,
            'include_abstract': True,
            'include_keywords_major': False,
            'include_keywords_minor': False,
        }

        assert (manager.cluster_names.columns == ['name']).all()
        assert len(manager.cluster_names) == 0

    def test_set_name(self) -> None:
        manager = ModelFactory.create(self.model, self.include_title, self.include_abstract, self.include_keywords_major, self.include_keywords_minor)
        manager.set_name(self.model_name)

        assert manager._name == self.model_name

    def test_construct_predictions_index_name(self) -> None:
        manager = ModelFactory.create(self.model, self.include_title, self.include_abstract, self.include_keywords_major, self.include_keywords_minor)
        manager.set_name(self.model_name)
        predictions_index_name = manager._construct_predictions_index_name(self.index)
        predictions_file_name = manager._construct_predictions_file_name(self.index)

        assert manager._name == self.model_name
        assert predictions_index_name == f'{self.index}_{manager._name}_predictions'
        assert predictions_file_name == f'{self.index}_predictions.json'

    def test_fit(self) -> None:
        manager = ModelFactory.create(self.model, self.include_title, self.include_abstract, self.include_keywords_major, self.include_keywords_minor)

        manager.fit(self.sample_data)

        assert len(manager.model.get_topics()) > 0

    def test_predict(self) -> None:
        manager = ModelFactory.create(self.model, self.include_title, self.include_abstract, self.include_keywords_major, self.include_keywords_minor)

        manager.fit(self.sample_data)
        predictions = manager.predict(self.sample_data)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (self.n_samples,)

    def test_transform(self) -> None:
        manager = ModelFactory.create(self.model, self.include_title, self.include_abstract, self.include_keywords_major, self.include_keywords_minor)

        manager.fit(self.sample_data)
        transformed = manager.transform(self.sample_data)

        assert isinstance(transformed, np.ndarray)
        assert transformed.shape[0] == self.n_samples

    def test_save(self) -> None:
        manager = ModelFactory.create(self.model, self.include_title, self.include_abstract, self.include_keywords_major, self.include_keywords_minor)

        manager.fit(self.sample_data)
        manager.save(self.model_name)

        assert os.path.exists(os.path.join(get_models_dir(), self.model_name, 'model.joblib'))
        assert os.path.exists(os.path.join(get_models_dir(), self.model_name, 'config.json'))
        assert os.path.exists(os.path.join(get_models_dir(), self.model_name, 'cluster_names.csv'))

    def test_save_only_model(self) -> None:
        manager = ModelFactory.create(self.model, self.include_title, self.include_abstract, self.include_keywords_major, self.include_keywords_minor)

        manager.fit(self.sample_data)
        manager.save(self.model_name, save_only='model')

        assert os.path.exists(os.path.join(get_models_dir(), self.model_name, 'model.joblib'))
        assert not os.path.exists(os.path.join(get_models_dir(), self.model_name, 'config.json'))
        assert not os.path.exists(os.path.join(get_models_dir(), self.model_name, 'cluster_names.csv'))

    def test_load(self) -> None:
        manager = ModelFactory.create(self.model, self.include_title, self.include_abstract, True, self.include_keywords_minor)

        manager.fit(self.sample_data)
        manager.save(self.model_name)

        manager_predictions = manager.predict(self.sample_data)
        manager_transformed = manager.transform(self.sample_data)

        loaded_manager = BERTopicManager.load(self.model_name)

        loaded_manager_predictions = loaded_manager.predict(self.sample_data)
        loaded_manager_transformed = loaded_manager.transform(self.sample_data)

        assert isinstance(loaded_manager, BERTopicManager)
        assert isinstance(loaded_manager.model, BERTopic)
        assert loaded_manager.random_state is None
        assert loaded_manager.preprocessing_args == {
            'include_title': True,
            'include_abstract': True,
            'include_keywords_major': True,
            'include_keywords_minor': False,
        }

        assert (loaded_manager.cluster_names.columns == ['name']).all()
        assert len(loaded_manager.cluster_names) > 0

        assert (loaded_manager_predictions == manager_predictions).all()
        assert (loaded_manager_transformed == manager_transformed).all()

    def test_load_from_factory(self) -> None:
        manager = ModelFactory.create(self.model, self.include_title, self.include_abstract, True, self.include_keywords_minor)

        manager.fit(self.sample_data)
        manager.save(self.model_name)

        loaded_manager = ModelFactory.load(self.model_name)

        assert isinstance(loaded_manager, BERTopicManager)
        assert isinstance(loaded_manager.model, BERTopic)
        assert loaded_manager.random_state is None
        assert loaded_manager.preprocessing_args == {
            'include_title': True,
            'include_abstract': True,
            'include_keywords_major': True,
            'include_keywords_minor': False,
        }

        assert (loaded_manager.cluster_names.columns == ['name']).all()
        assert len(loaded_manager.cluster_names) > 0

    def test_evalutate(self) -> None:
        manager = ModelFactory.create(self.model, self.include_title, self.include_abstract, self.include_keywords_major, self.include_keywords_minor)

        manager.fit(self.sample_data)
        metrics = manager.evaluate(self.sample_data)

        assert isinstance(metrics, pd.DataFrame)
        assert 'Silhouette' in metrics.columns
        assert 'Calinski-Harabasz' in metrics.columns
        assert 'Davies-Bouldin' in metrics.columns
        assert len(metrics) == 1

    def test_precompute_predictions_max_new_predictions_amount_limited(self) -> None:
        manager = ModelFactory.create(self.model, self.include_title, self.include_abstract, self.include_keywords_major, self.include_keywords_minor)

        es = get_es_client()

        manager.fit(self.sample_data)
        manager.set_name(self.model_name)

        manager.precompute_predictions(self.index, max_new_predictions=100, batch_size=10, es=es)
        predictions_index_name = manager._construct_predictions_index_name(self.index)

        # Refresh the index
        es.indices.refresh(index=predictions_index_name)

        assert es.indices.exists(index=predictions_index_name)
        assert es.count(index=predictions_index_name)['count'] == 100

        # Check if the mapping is correct
        mapping = es.indices.get_mapping(index=predictions_index_name)[predictions_index_name]['mappings']['properties']
        assert mapping['date']['type'] == 'date'
        assert mapping['label']['type'] == 'integer'
        assert mapping['PMID']['type'] == 'keyword'

        manager.clear_predictions(self.index, es)

        assert not es.indices.exists(index=predictions_index_name)

    def test_precompute_predictions_max_predictions_time_limited(self) -> None:
        manager = ModelFactory.create(self.model, self.include_title, self.include_abstract, self.include_keywords_major, self.include_keywords_minor)

        es = get_es_client()

        manager.fit(self.sample_data)
        manager.set_name(self.model_name)

        manager.precompute_predictions(self.index, timeout=5, batch_size=10, es=es)
        predictions_index_name = manager._construct_predictions_index_name(self.index)

        # Refresh the index
        es.indices.refresh(index=predictions_index_name)

        assert es.indices.exists(index=predictions_index_name)
        assert es.count(index=predictions_index_name)['count'] > 0

        # Check if the mapping is correct
        mapping = es.indices.get_mapping(index=predictions_index_name)[predictions_index_name]['mappings']['properties']
        assert mapping['date']['type'] == 'date'
        assert mapping['label']['type'] == 'integer'
        assert mapping['PMID']['type'] == 'keyword'

        manager.clear_predictions(self.index, es)

        assert not es.indices.exists(index=predictions_index_name)
