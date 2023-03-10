# Using the CLI: Minimal example

In this example, we guide you through the end-to-end process of using `pmtrendviz` to visualize trends in the biomedical literature.

## Table of Contents

- [Using the CLI: Minimal example](#using-the-cli-minimal-example)
  - [Table of Contents](#table-of-contents)
  - [1. Importing the data](#1-importing-the-data)
  - [2. Listing available models](#2-listing-available-models)
  - [3. Training a model](#3-training-a-model)
  - [4. Precomputing the predictions](#4-precomputing-the-predictions)


## 1. Importing the data

The first step is to import the data into Elasticsearch, which can be done using the `import` command.

To download and import the latest 100 (from 1166 in total) files from PubMed Baseline, run the following command:

```bash
pmtrendviz import -x pubmed -n 100
```

and confirm the prompt. By default, all data will be imported.

You may increase the number of threads with `-t` to speed up the extraction process, however, beyond `-t 4`, the bottleneck will probably be the download speed.

## 2. Listing available models

After importing the data, you have two options to create a model for the visualization, either by training your own model or by using one of the pre-trained models.

To list the available trainable and pre-trained models, run the following command:

```bash
pmtrendviz list --trainable --pretrained
```

## 3. Training a model

To train a simple TF-IDF-based model on a random sample of 500k abstracts, run the following command:

```bash
pmtrendviz train -m tfidf_truncatedsvd_kmeans -x pubmed -n 500000 -s my-tfidf-model -p uniform
```

The trained model will be saved to the `models` directory.

Alternatively, you can install a pre-trained model from Huggingface, for example by running the following command:

```bash
pmtrendviz install tfidf-3m-100
```

## 4. Precomputing the predictions

Since the visualization of the trends requires a large number of rather expensive predictions, we need to precompute the predictions with the `precompute` command:

```bash
pmtrendviz precompute -x pubmed -m my-tfidf-model -b 10000
```

By default, this will precompute the predictions for all documents in the index, but you can also specify a maximum number of new predictions to make with `-P` or a maximum number of seconds to run with `-T`. You can also specify the order in which to predict the cluster of the abstracts with `-p`, which can be either `uniform` or `forward`. The `forward` option is faster, but will only prdocue a meaningful visualization if run on the whole dataset, while the `uniform` option is slower, but will produce a meaningful visualization even if run on a smaller subset of the data.