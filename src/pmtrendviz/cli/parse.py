from argparse import ArgumentParser


def parse_import_end_to_end_arguments(parser: ArgumentParser | None = None) -> ArgumentParser:
    """
    Parse the command line arguments

    Parameters
    ----------
    parser : argparse.ArgumentParser | None, optional
        The parser to add the arguments to, by default None

    Returns
    -------
    argparse.ArgumentParser
        The parser with the added arguments
    """

    if parser is None:
        parser = ArgumentParser(description='Download the PubMed data')

    parser.add_argument('-t', '--n-threads', type=int, default=1, help='The number of threads to use for downloading the files and extracting the data')
    parser.add_argument('-n', '--last-n-files', type=int, default=None, help='The number of files to import, starting from the most recent file')
    parser.add_argument('-x', '--index', required=True, type=str, help='The name of the index to import the data into')
    parser.add_argument('-R', '--max-retries', type=int, default=3, help='The maximum number of times to retry downloading a file if it fails')
    parser.add_argument('-B', '--backoff-factor', type=float, default=300.0, help='The backoff factor for the exponential backoff strategy')
    parser.add_argument('-y', '--yes', action='store_true', help='Do not ask for confirmation before importing the files')

    return parser


def parse_download_arguments(parser: ArgumentParser | None = None) -> ArgumentParser:
    """
    Parse the arguments for the download script

    Parameters
    ----------
    parser : argparse.ArgumentParser | None, optional
        The parser to add the arguments to. If None, a new parser is created

    Returns
    -------
    parser : argparse.ArgumentParser
        The parser with the added arguments
    """

    if parser is None:
        parser = ArgumentParser(description='Download the PubMed data from the online directory')

    parser.add_argument('-t', '--n-threads', type=int, default=1, help='The number of threads to use for downloading the files')
    parser.add_argument('-n', '--last-n-files', type=int, default=None, help='The number of files to download, starting from the most recent file')
    parser.add_argument('-f', '--overwrite', action='store_true', help='Overwrite the existing files')
    parser.add_argument('-o', '--target-dir', type=str, required=True, help='The directory to save the downloaded and any temporary files')
    parser.add_argument('-R', '--max-retries', type=int, default=2, help='The maximum number of times to retry downloading a file if it fails')
    parser.add_argument('-B', '--backoff-factor', type=float, default=300.0, help='The backoff factor for the exponential backoff strategy')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more information about the download progress')
    parser.add_argument('-y', '--yes', action='store_true', help='Skip the confirmation prompt')

    return parser


def parse_extract_arguments(parser: ArgumentParser | None = None) -> ArgumentParser:
    """
    Parse the command line arguments

    Parameters
    ----------
    parser : argparse.ArgumentParser | None, optional
        The parser to add the arguments to, by default None

    Returns
    -------
    argparse.ArgumentParser
        The parser with the added arguments
    """
    if parser is None:
        parser = ArgumentParser(description='Extract the relevant information from the PubMed XML files and store it in json files')

    parser.add_argument('-t', '--n-threads', type=int, default=1, help='The number of threads to use for extracting data from the files')
    parser.add_argument('-f', '--overwrite', action='store_true', help='Overwrite the existing files')
    parser.add_argument('-s', '--src-dir', type=str, required=True, help='Source directory of xml-files')
    parser.add_argument('-o', '--target-dir', type=str, required=True, help='Target directory of json files with extracted data')
    parser.add_argument('-y', '--yes', action='store_true', help='Skip the confirmation prompt')

    return parser


def parse_index_arguments(parser: ArgumentParser | None = None) -> ArgumentParser:
    """
    Parse the command line arguments

    Parameters
    ----------
    parser : argparse.ArgumentParser | None, optional
        The parser to add the arguments to, by default None

    Returns
    -------
    argparse.ArgumentParser
        The parser with the added arguments
    """
    if parser is None:
        parser = ArgumentParser(description='Import the extracted data into an Elasticsearch index')

    parser.add_argument('-n', '--last-n-files', type=int, default=None)
    parser.add_argument('-s', '--src-dir', type=str)
    parser.add_argument('-x', '--index', required=True, type=str, help='The name of the index to import the data into')
    parser.add_argument('-y', '--yes', action='store_true', help='Skip the confirmation prompt')

    return parser


def parse_train_arguments(parser: ArgumentParser | None = None) -> ArgumentParser:
    """
    Parse the command line arguments

    Parameters
    ----------
    parser : argparse.ArgumentParser | None, optional
        The parser to add the arguments to

    Returns
    -------
    argparse.ArgumentParser
        The parser with the added arguments
    """
    if parser is None:
        parser = ArgumentParser(description='Training for the PubMed Topic Clustering and Trend Visualization')

    # General arguments
    parser.add_argument('-s', '--save', type=str, required=True, help='The path to save the model to')
    parser.add_argument('-f', '--overwrite', type=bool, default=False, help='Overwrite the model if it already exists')

    # Data arguments
    parser.add_argument('-x', '--index', type=str, required=True, help='The path to the index')

    # Text combination arguments
    parser.add_argument('-n', '--n-samples', type=int, default=int(1e6), help='The number of samples to use for training')
    parser.add_argument('-p', '--sample-method', type=str, default='uniform', help='The method to use for sampling the data. Can be "uniform", "forward" or "backward"')
    parser.add_argument('-r', '--random-state', type=int, default=None, help='The random state to use')

    # Preprocessing arguments
    parser.add_argument('--include-title', type=bool, default=True, help='Include the title in the training data')
    parser.add_argument('--include-abstract', type=bool, default=True, help='Include the abstract in the training data')
    parser.add_argument('--include-keywords-major', type=bool, default=False, help='Include the major keywords in the training data')
    parser.add_argument('--include-keywords-minor', type=bool, default=False, help='Include the minor keywords in the training data')

    # Model arguments
    parser.add_argument('-m', '--model', type=str, default='tfidf_truncatedsvd_kmeans', help='The model to train')
    parser.add_argument('--stop-words', type=str, default='english', help='The stop words to use, if applicable')
    parser.add_argument('--max-df', type=float, default=0.999, help='The maximum document frequency, if applicable')
    parser.add_argument('--min-df', type=float, default=0.001, help='The minimum document frequency, if applicable')
    parser.add_argument('--n-components', type=int, default=100, help='The number of components to use for the SVD, if applicable')
    parser.add_argument('--n-clusters', type=int, default=100, help='The number of clusters to use for the KMeans, if applicable')
    parser.add_argument('--ngram-range', type=int, nargs=2, default=(1, 1), help='The n-gram range to use for the TF-IDF, if applicable')

    # Spacy arguments
    parser.add_argument('--spacy-model', type=str, default='en_core_sci_sm', help='The spacy model to use for preprocessing, if applicable')
    parser.add_argument('--spacy-disable', type=str, nargs='+', default=None, help='The spacy components to disable, if applicable')

    return parser


def parse_precompute_arguments(parser: ArgumentParser | None = None) -> ArgumentParser:
    """
    Parse the command line arguments for precomputing the predictions

    Parameters
    ----------
    parser : argparse.ArgumentParser | None, optional
        The parser to add the arguments to

    Returns
    -------
    argparse.ArgumentParser
        The parser with the added arguments
    """
    if parser is None:
        parser = ArgumentParser(description='Precompute predictions for the PubMed Topic Clustering and Trend Visualization')

    parser.add_argument('-x', '--index', type=str, required=True, help='The name of the index in Elasticsearch')
    parser.add_argument('-m', '--model-name', type=str, required=True, help='The name of the model use for the prediction')
    parser.add_argument('-P', '--max-new-predictions', type=int, default=None, help='The maximum number of new predictions to make before stopping')
    parser.add_argument('-T', '--timeout', type=int, default=None, help='The maximum number of seconds to run before stopping')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='The number of documents to process at a time (Higher values are faster, but use more memory)')
    parser.add_argument('-p', '--sample-method', type=str, default='uniform', help='The method to use for sampling the data. Can be "uniform" or "forward"')

    return parser


def parse_app_arguments(parser: ArgumentParser | None = None) -> ArgumentParser:
    """
    Parse the command line arguments

    Parameters
    ----------
    parser : argparse.ArgumentParser | None, optional
        The parser to add the arguments to

    Returns
    -------
    argparse.ArgumentParser
        The parser with the added arguments
    """
    if parser is None:
        parser = ArgumentParser(description='Frontend for the PubMed Topic Clustering and Trend Visualization')

    parser.add_argument('-x', '--index', type=str, default='pubmed', help='The name of the index in Elasticsearch')

    return parser


def parse_list_arguments(parser: ArgumentParser | None = None) -> ArgumentParser:
    """
    Parse the command line arguments for listing the available models and managers

    Parameters
    ----------
    parser : argparse.ArgumentParser | None, optional
        The parser to add the arguments to

    Returns
    -------
    ArgumentParser
        The parser with the added arguments
    """
    if parser is None:
        parser = ArgumentParser(description='List available models and managers for the PubMed Topic Clustering and Trend Visualization')

    parser.add_argument('--trainable', action='store_true', help='List the trainable models')
    parser.add_argument('--saved', action='store_true', help='List the saved models')
    parser.add_argument('--managers', action='store_true', help='List the implemented managers')
    parser.add_argument('--pretrained', action='store_true', help='List the installable pretrained models')
    parser.add_argument('--with-predictions', action='store_true', help='List the saved models with predictions available')

    return parser


def parse_install_pretrained_arguments(parser: ArgumentParser | None = None) -> ArgumentParser:
    """
    Parse the command line arguments for installing a pretrained model

    Parameters
    ----------
    parser : argparse.ArgumentParser | None, optional
        The parser to add the arguments to

    Returns
    -------
    ArgumentParser
        The parser with the added arguments
    """
    if parser is None:
        parser = ArgumentParser(description='Install a pretrained model for the PubMed Topic Clustering and Trend Visualization')

    # Don't require the double dash
    # The user should be able to use `pmtrendviz install model_name`
    parser.add_argument('model_name', type=str, help='The name of the model to install')
    parser.add_argument('-f', '--overwrite', action='store_true', help='Overwrite the model if it already exists')

    return parser


def parse_remove_arguments(parser: ArgumentParser | None = None) -> ArgumentParser:
    """
    Parse the command line arguments for removing a model

    Parameters
    ----------
    parser : argparse.ArgumentParser | None, optional
        The parser to add the arguments to

    Returns
    -------
    ArgumentParser
        The parser with the added arguments
    """
    if parser is None:
        parser = ArgumentParser(description='Remove a model from the PubMed Topic Clustering and Trend Visualization')

    # Don't require the double dash
    # The user should be able to use `pmtrendviz remove model_name`
    parser.add_argument('model_name', type=str, help='The name of the model to remove')
    parser.add_argument('-i', '--ignore-errors', action='store_true', help='Ignore errors when removing the model')

    return parser
