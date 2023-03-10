# Using the python API: Minimal example

This example shows how to use the `pmtrendviz` package in your own python code.

## 1. Use a pretrained model
```python
import pmtrendviz as pm

# Install a pretrained model
pm.install('test_model')

# Load the model
manager = pm.load('test_model')

# Predict a cluster for a given query
manager.predict('example query')

# Access the model directly
manager.model
```

## 2. Train a model
```python
import pmtrendviz as pm
from pmtrendviz.train.data import sample_training_data

# Create a new model
manager = pm.create(
    model='tfidf_truncatedsvd_kmeans',  # Name of the factory method
    n_components=100,
    n_clusters=100,
    n_gram_range=(1, 2)
)

# Sample training data
data = sample_training_data(index='pubmed', n_samples=100_000, method='uniform')

# Train the model
manager.fit(data['text'])

# Save the model
manager.save('example_model')
```