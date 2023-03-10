import os

import pandas as pd
import pytest

import pmtrendviz as pm
from pmtrendviz.train.data import sample_training_data
from pmtrendviz.utils import get_es_client, get_models_dir


@pytest.mark.usefixtures('manage_pytest_index', 'manage_prediction_indices')
class TestInit:
    sample_data: pd.DataFrame = pd.DataFrame()
    n_samples: int = 100
    index: str = 'pytest'
    model_name: str = 'pytest-model'
    install_model_name: str = 'test-model'
    model: str = 'tfidf_truncatedsvd_kmeans'

    @classmethod
    def setup_class(cls) -> None:
        # Clear the test model directory
        pm.remove(cls.model_name, ignore_errors=True)
        pm.remove(cls.install_model_name, ignore_errors=True)

        # Sample some data
        cls.sample_data = sample_training_data(cls.index, random_state=42, n_samples=cls.n_samples, method='forward')['text']

    @classmethod
    def teardown_class(cls) -> None:
        # Clear the test model directory
        pm.remove(cls.model_name, ignore_errors=True)
        pm.remove(cls.install_model_name, ignore_errors=True)

    def teardown_method(self) -> None:
        # Clear the test model directory
        pm.remove(self.model_name, ignore_errors=True)
        pm.remove(self.install_model_name, ignore_errors=True)

    def test_create_remove(self) -> None:
        """Test the create and remove functions."""
        # Create a model
        manager = pm.create('tfidf_truncatedsvd_kmeans', include_title=True, include_abstract=True, include_keywords_major=True, include_keywords_minor=True)

        assert isinstance(manager, pm.ModelManager)

        # Fit the model
        manager.fit(self.sample_data)

        manager.set_name(self.model_name)

        # Precompute the predictions
        manager.precompute_predictions(self.index, max_new_predictions=1000, batch_size=100, method='forward')

        # Save the model
        manager.save(self.model_name)

        # Remove the model
        pm.remove(self.model_name)

    def test_install_load_remove(self) -> None:
        """Test the install and remove functions."""
        # Install a model
        pm.install(self.install_model_name)

        assert os.path.exists(os.path.join(get_models_dir(), self.install_model_name))
        assert os.path.exists(os.path.join(get_models_dir(), self.install_model_name, 'model.joblib'))
        assert os.path.exists(os.path.join(get_models_dir(), self.install_model_name, 'cluster_names.csv'))
        assert os.path.exists(os.path.join(get_models_dir(), self.install_model_name, 'config.json'))

        # Load the model
        manager = pm.load(self.install_model_name)

        assert isinstance(manager, pm.ModelManager)

        del manager

        # Remove the model
        pm.remove(self.install_model_name)

        assert not os.path.exists(os.path.join(get_models_dir(), self.install_model_name))

        # Check that the prediction indices have been removed
        es = get_es_client()
        assert len(es.indices.get(index=f'*_{self.install_model_name}_predictions')) == 0
