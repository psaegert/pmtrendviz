"""The command line interface for the PubMed Topic Clustering and Trend Visualization tool."""

import argparse
import os

from ..utils import get_models_dir
from . import parse


def run_cli() -> argparse.Namespace:
    """
    Run the command line interface

    Returns
    -------
    argparse.Namespace
        The parsed arguments
    """
    # Create the parser
    parser = argparse.ArgumentParser(description='PubMed Topic Clustering and Trend Visualization')
    parser.add_argument('-l', '--log-level', type=str, default='WARNING', help='The log level to use')

    # Add the subparsers
    subparsers = parser.add_subparsers(dest='command')

    # Dependent on the subcommand, add the corresponding arguments

    # --- Import ---

    # Import subcommand
    import_parser = subparsers.add_parser('import', help='Import the PubMed data')
    import_parser = parse.parse_import_end_to_end_arguments(import_parser)

    # Import download subcommand
    import_download_parser = subparsers.add_parser('import-download', help='Download the PubMed data from the online directory')
    import_download_parser = parse.parse_download_arguments(import_download_parser)

    # Import extract subcommand
    import_extract_parser = subparsers.add_parser('import-extract', help='Extract the relevant information from the PubMed XML files and store it in json files')
    import_extract_parser = parse.parse_extract_arguments(import_extract_parser)

    # Import index subcommand
    import_index_parser = subparsers.add_parser('import-index', help='Import the extracted data into an Elasticsearch index')
    import_index_parser = parse.parse_index_arguments(import_index_parser)

    # --- Train ---

    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser = parse.parse_train_arguments(train_parser)

    # --- Precompute ---

    # Precompute subcommand
    precompute_parser = subparsers.add_parser('precompute', help='Precompute the predictions for the app')
    precompute_parser = parse.parse_precompute_arguments(precompute_parser)

    # --- Install pretrained ---

    # Install pretrained subcommand
    install_pretrained_parser = subparsers.add_parser('install', help='Install a pretrained model')
    install_pretrained_parser = parse.parse_install_pretrained_arguments(install_pretrained_parser)

    # --- Remove ---
    remove_parser = subparsers.add_parser('remove', help='Remove a model and its predictions')
    remove_parser = parse.parse_remove_arguments(remove_parser)

    # --- List ---

    # List subcommand
    list_parser = subparsers.add_parser('list', help='List the available models')
    list_parser = parse.parse_list_arguments(list_parser)

    # Parse the arguments
    args = parser.parse_args()

    # TODO: Implement the default mode
    if args.command is None:
        print('No command specified. Running in default mode.')
        raise NotImplementedError

    # When training, decide what to do when the model already exists
    if args.command == 'train' and not args.overwrite:
        # Check if the model already exists
        if os.path.exists(os.path.join(get_models_dir(), args.save)):
            # Append a number to the model name
            i = 1
            while os.path.exists(os.path.join(get_models_dir(), f'{args.save}_{i}')):
                i += 1
            args.save = f'{args.save}_{i}'

    return args
