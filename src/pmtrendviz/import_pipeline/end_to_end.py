import argparse
import atexit
import gzip
import logging
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

# Import the local modules
from ..utils import ImportOptimizer, MaxRetriesExceeded, UserAbort, download_file, get_es_client
from .download import PUBMED_ONLINE_DATA_DIR, fetch_files, filter_gz_filenames, \
    parse_html_directory, verify_md5
from .extract import extract_worker
from .index import create_index, import_file

logger = logging.getLogger('pmtrendviz.import_pipeline.end_to_end')


def import_pipeline(file_url: str, file_path_compressed: str, max_retries: int = 3, backoff_factor: float = 300.0) -> str | None:
    """
    Download, verify, and uncompress a file from the online directory.
    Extract the relevant information from the PubMed XML files and store it in json files.
    Import the extracted data into an Elasticsearch index.

    Parameters
    ----------
    file_url : str
        The url of the file to download
    file_path_compressed : str
        The path to save the compressed file to
    max_retries : int, optional
        The maximum number of retries, by default 3
    backoff_factor : float, optional
        The backoff factor for exponential backoff, by default 300

    Returns
    -------
    str
        The name of the resulting json file
    """

    # If any of the folders up to the file path do not exist, create them
    os.makedirs(os.path.dirname(file_path_compressed), exist_ok=True)

    file_path_uncompressed = file_path_compressed[:-3]  # Remove the .gz extension
    file_path_uncompressed_temp = file_path_uncompressed + '.tmp'

    file_path_json = file_path_uncompressed.replace('.xml', '.json')
    file_path_json_temp = file_path_json + '.tmp'

    file_path_compressed_md5 = file_path_compressed + '.md5'

    # Remove all temporary files associated with this file
    for file_path in [file_path_uncompressed_temp, file_path_json_temp]:
        if os.path.exists(file_path):
            os.remove(file_path)

    # Remove all files associated with this file
    for file_path in [file_path_json, file_path_uncompressed, file_path_compressed, file_path_compressed_md5]:
        if os.path.exists(file_path):
            os.remove(file_path)

    # Download the compressed file
    try:
        download_file(file_url, file_path_compressed, max_retries=max_retries, backoff_factor=backoff_factor)
    except MaxRetriesExceeded:
        logger.error(f'Could not download file: {file_url}')
        return None

    # Download the md5 file
    try:
        download_file(file_url + '.md5', file_path_compressed_md5, max_retries=max_retries, backoff_factor=backoff_factor)
    except MaxRetriesExceeded:
        logger.error(f'Could not download file: {file_url + ".md5"}')
        return None

    # Verify the md5
    if not verify_md5(file_path_compressed):
        logger.error(f'Could not verify md5 for file: {file_path_compressed}')
        return None

    # Uncompress the file
    with gzip.open(file_path_compressed, 'rb') as f_in:
        with open(file_path_uncompressed_temp, 'wb') as f_out:
            f_out.write(f_in.read())

    # Rename the temporary file to the final file
    os.rename(file_path_uncompressed_temp, file_path_uncompressed)

    # Remove the compressed file
    os.remove(file_path_compressed)

    # Remove the md5 file
    os.remove(file_path_compressed + '.md5')

    # Extract the relevant data from the XML file and store it in a json file
    extract_worker(xml_file_path=file_path_uncompressed, json_file_path=file_path_json_temp)

    # Rename the temporary file to the final file
    os.rename(file_path_json_temp, file_path_json)

    # Remove the xml file
    os.remove(file_path_uncompressed)

    return file_path_json


def import_end_to_end(args: argparse.Namespace) -> None:
    """
    Run the import pipeline for a specified number of files.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the program

    Returns
    -------
    None
    """
    # Display a warning if the number of threads is greater than 2
    if args.n_threads > 2:
        logger.warning(f'Using more than 2 threads (current: {args.n_threads}) may cause problems with the PubMed servers')
        # Also print the warning to stderr
        print(f'WARNING: Using more than 2 threads (current: {args.n_threads}) may cause problems with the PubMed servers', file=sys.stderr)

    # Create a temporary directory to store the downloaded files
    tmp_dir = tempfile.mkdtemp()

    # Create an import optimizer instance
    import_optimizer = ImportOptimizer(index=args.index)

    logger.debug(f"Temporary directory: {tmp_dir}")

    # Register a function to remove the temporary directory when the program exits
    atexit.register(lambda: shutil.rmtree(tmp_dir, ignore_errors=True))

    # --- Fetch filenames ---

    # Fetch the filenames from the online directory
    html = fetch_files(PUBMED_ONLINE_DATA_DIR)
    logger.info('Fetched online directory')

    # Parse the html directory to get all file names
    file_names = parse_html_directory(html)
    logger.info(f'Parsed {len(file_names)} links from the online directory')

    # Filter the filenames to only get the .gz filenames
    filtered_file_names = filter_gz_filenames(file_names)
    logger.info(f'Filtered {len(filtered_file_names)} suitable files within the online directory')

    # Fetch the last filenames
    if args.last_n_files is not None:
        # Only keep the last n files
        filtered_file_names = filtered_file_names[-args.last_n_files:]

    # Only import files that have not been imported yet
    filtered_file_names = import_optimizer.filter_registered_files(filtered_file_names)

    # As the user if they want to continue
    if not args.yes:
        answer = input(f'Importing {len(filtered_file_names)} files. Continue? (y/n)')
        if answer.lower() not in ['y', 'yes']:
            logger.error('User aborted import')
            raise UserAbort("Confirmation of import end-to-end")

    # --- Execute the Import Pipeline ---

    es = get_es_client()

    create_index(es, index=args.index)

    # Create a progress bar
    pbar = tqdm(total=len(filtered_file_names))
    pbar.set_description('Importing files')

    # Create a workload
    files_to_import = filtered_file_names

    with ProcessPoolExecutor(max_workers=args.n_threads) as executor:
        futures = []
        for _ in range(min(args.n_threads, len(files_to_import))):
            # Get the next file to import
            file_name = files_to_import.pop()

            # Submit the import pipeline to the executor
            futures.append(executor.submit(
                import_pipeline,
                file_url=PUBMED_ONLINE_DATA_DIR + file_name,
                file_path_compressed=os.path.join(tmp_dir, file_name),
                max_retries=args.max_retries,
                backoff_factor=args.backoff_factor))

        while files_to_import:
            logger.debug('Waiting for futures to complete')

            for future in as_completed(futures, timeout=args.backoff_factor * (2 ** args.max_retries)):
                # Get the result from the future
                file_path_json = future.result()

                logger.debug(f'Future {future} result: {file_path_json}')

                # Check if the download was successful
                if file_path_json is None:
                    # Download failed, skip the file
                    logger.error(f'Download of file {future} failed.')
                elif not os.path.exists(file_path_json):
                    # Download successful, but the json file was not created
                    logger.error(f'Download of file {future} failed. JSON file not created.')
                else:
                    # Download successful, import the json file into Elasticsearch
                    logger.debug(f'Importing {file_path_json} to Elasticsearch index {args.index}')
                    import_file(file_path_json, client=es, index=args.index)

                    # Register the file as imported
                    import_optimizer.register_imported_file(file_path_json)

                    # Remove the json file
                    if os.path.exists(file_path_json):
                        os.remove(file_path_json)

                # Remove the future from the list
                futures.remove(future)

                # Submit the next file
                if files_to_import:
                    # Get the next file to import
                    file_name = files_to_import.pop()

                    logger.debug(f'Submitting {file_name} to executor')

                    # Submit the import pipeline to the executor
                    futures.append(executor.submit(
                        import_pipeline,
                        file_url=PUBMED_ONLINE_DATA_DIR + file_name,
                        file_path_compressed=os.path.join(tmp_dir, file_name),
                        max_retries=args.max_retries,
                        backoff_factor=args.backoff_factor))

                # Update the progress bar
                pbar.update(1)

                time.sleep(0.1)

            time.sleep(0.1)

    # Wait for all futures to complete
    for future in as_completed(futures, timeout=args.backoff_factor * (2 ** args.max_retries)):
        # Get the result from the future
        file_path_json = future.result()

        logger.debug(f'Future {future} result: {file_path_json}')

        # Check if the download was successful
        if file_path_json is None:
            # Download failed, skip the file
            logger.error(f'Download of file {future} failed.')
        elif not os.path.exists(file_path_json):
            # Download successful, but the json file was not created
            logger.error(f'Download of file {future} failed. JSON file not created.')
        else:
            # Download successful, import the json file into Elasticsearch
            logger.debug(f'Importing {file_path_json} to Elasticsearch index {args.index}')
            import_file(file_path_json, client=es, index=args.index)

            # Register the file as imported
            import_optimizer.register_imported_file(file_path_json)

            # Remove the json file
            if os.path.exists(file_path_json):
                os.remove(file_path_json)

        # Remove the future from the list
        futures.remove(future)

        pbar.update(1)

        time.sleep(0.1)

    pbar.set_postfix_str('Done')
    pbar.close()
