"""Entry point for the PubMed Topic Clustering and Trend Visualization application."""

import logging
import os
from logging.handlers import TimedRotatingFileHandler

# Import the functions from the submodules
from .cli.cli import run_cli

LOG_FILE = 'logs/main.log'
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[TimedRotatingFileHandler(LOG_FILE, when='midnight', backupCount=7)])
logger = logging.getLogger('pmtrendviz.__main__')


def main() -> None:
    args = run_cli()

    # Set the log level for all loggers
    logging.getLogger('pmtrendviz').setLevel(args.log_level)
    logger.debug(f'Program called with following arguments: {vars(args)}')

    match args.command:
        case 'import':
            from .import_pipeline.end_to_end import import_end_to_end
            import_end_to_end(args)
        case 'import-download':
            from .import_pipeline.download import download
            download(args)
        case 'import-extract':
            from .import_pipeline.extract import extract
            extract(args)
        case 'import-index':
            from .import_pipeline.index import index
            index(args)
        case 'train':
            from .train.train import train
            train(args)
        case 'precompute':
            from .precompute.precompute import precompute
            precompute(args)
        case 'list':
            from .list.list import list_
            list_(args)
        case 'install':
            from .models.factory import ModelFactory
            ModelFactory.install(args.model_name, args.overwrite)
        case 'remove':
            from .models.factory import ModelFactory
            ModelFactory.remove(args.model_name, args.ignore_errors)
        case _:
            if args.command is None:
                raise ValueError('No command specified')
            raise ValueError(f'Unknown command: {args.command}')


if __name__ == '__main__':
    main()
