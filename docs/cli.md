# `pmtrendviz` CLI Documentation

The entry point for the project is the `main.py` script.

Option 1:<br>
When installed with pip, it can be executed with

```bash
pmtrendviz [options]
```

Option 2:<br>
You can also run the script directly via

```bash
python main.py [options]
```


## Commands

**Note:** Unless you know what you are doing, we recommend using `-x pubmed` as the index name. This will ensure that the index is compatible with the visualization.

### Import

The `import` command is used to import the data into ElasticSearch. Running the whole import pipeline can be done with the following command:

```bash
pmtrendviz import [options]
```
- `-x`, `--index`: The name of the index to import the data into
- `-n`, `--last-n-files`: The number of files to import, starting from the most recent file
- `-t`, `--n-threads`: The number of threads to use for downloading the files and extracting the data
- `-B`, `--backoff-factor`: The backoff factor to use for exponential backoff
- `-R`, `--max-retries`: The maximum number of times to retry downloading a file if it fails
- `-y`, `--yes`: Skip the confirmation prompt

You may also run the individual steps of the import pipeline separately:

### Import-Download

Downloads the XML files from the PubMed server into a target directory.

```bash
pmtrendviz import-download [options]
```

The `import-download` command has the following options:
- `-o`, `--target-dir`: The directory to save the downloaded and any temporary files
- `-n`, `--last-n-files`: The number of files to download, starting from the most recent file
- `-f`, `--overwrite`: Overwrite the existing files
- `-t`, `--n-threads`: The number of threads to use for downloading the files
- `-R`, `--max-retries`: The maximum number of times to retry downloading a file if it fails
- `-B`, `--backoff-factor`: The backoff factor to use for exponential backoff
- `-v`, `--verbose`: Print more information about the download progress
- `-y`, `--yes`: Skip the confirmation prompt

### Import-Extract

Extracts the data from the downloaded XML files into json files.

```bash
pmtrendviz import-extract [options]
```

- `-s`, `--src-dir`: The directory to read the downloaded XML files from
- `-o`, `--target-dir`: The directory to save the json files with the extracted data to
- `-f`, `--overwrite`: Overwrite the existing json files
- `-n`, `--n-threads`: The number of threads to use for extracting data from the files
- `-y`, `--yes`: Skip the confirmation prompt

### Import-Index

Indexes the extracted data into ElasticSearch.

```bash
pmtrendviz import-index [options]
```

- `-s`, `--src-dir`: The directory to read the prepared json files with the extracted data from
- `-x`, `--index`: The name of the index to import the data into
- `-n`, `--last-n-files`: The number of files to index, starting from the most recent file
- `-y`, `--yes`: Skip the confirmation prompt

---

### Train

The `train` command is used to train models.

```bash
pmtrendviz train [options]
```

- `-s`, `--save`: The path to save the model to
- `-f`, `--overwrite`: Overwrite the model if it already exists
- `-x`, `--index`: The path to the index
- `-n`, `--n-samples`: The number of samples to use for training
- `-p`, `--sample-method`: The method to use for sampling the data. Can be "uniform", "forward" or "backward"
- `-r`, `--random-state`: The random state to use
- `--include-title`: Include the title in the training data
- `--include-abstract`: Include the abstract in the training data
- `--include-keywords-major`: Include the major keywords in the training data
- `--include-keywords-minor`: Include the minor keywords in the training data
- `-m`, `--model`: The model to train
- `--stop-words`: The stop words to use, if applicable
- `--max-df`: The maximum document frequency, if applicable
- `--min-df`: The minimum document frequency, if applicable
- `--n-components`: The number of components to use for the SVD, if applicable
- `--n-clusters`: The number of clusters to use for the KMeans, if applicable
- `--ngram-range`: The n-gram range to use for the TF-IDF, if applicable
- `--spacy-model`: The spacy model to use for preprocessing, if applicable
- `--spacy-disable`: The spacy components to disable, if applicable

---

### Precompute Predictions

The `precompute` command is used to precompute the predictions for the visualization.

```bash
pmtrendviz precompute [options]
```

- `-x`, `--index`: The name of the source index in Elasticsearch
- `-m`, `--model-name`: The name of the model to use for the prediction
- `-P`, `--max-new-predictions`: The maximum number of new predictions to make before stopping
- `-T`, `--timeout`: The maximum number of seconds to run before stopping
- `-b`, `--batch-size`: The number of documents to process at a time (Higher values are faster, but use more memory)
- `-p`, `--sample-method`: The method to use for sampling the data. Can be "uniform" or "forward"

---

### List

The `list` command is used to list the available models.

```bash
pmtrendviz list [options]
```

- `--trainable`: List the trainable models
- `--saved`: List the saved models
- `--managers`: List the implemented managers
- `--pretrained`: List the installable pretrained models
- `--with-predictions`: List the saved models with predictions available

---

### Install

The `install` command is used to install pretrained `pmtrendviz` models from Huggingface.

```bash
pmtrendviz install [options]
```

- `model_name`: The name of the model to install

---

### Remove

The `remove` command is used to remove models and their predictions.

```bash
pmtrendviz remove [options]
```

- `model_name`: The name of the model to remove
- `-i`, `--ignore-errors`: Ignore errors when removing the model