import argparse
import gzip
import hashlib
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import requests
from tqdm import tqdm

from ..utils import MaxRetriesExceeded, UserAbort, download_file

PUBMED_ONLINE_DATA_DIR = 'https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/'

logger = logging.getLogger('pmtrendviz.import_pipeline.download')


def fetch_files(url: str) -> str:
    '''
    Fetch the files from the online directory

    Returns
    -------
    response : str
    '''
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        logger.error(f'Could not fetch files from online data directory: Error {response.status_code}')
        raise Exception(f'Could not fetch files from online data directory: Error {response.status_code}')


def parse_html_directory(response: str) -> list:
    '''
    Parse the html directory to get all file names

    Parameters
    ----------
    response : str
        The html response from the online directory

    Return
    -------
    file_names : list
    '''
    import re
    pattern = re.compile(r'href="(.+?)"')
    return re.findall(pattern, response)


def filter_gz_filenames(file_names: list) -> List[str]:
    '''
    Filter the file names to only get the .gz files

    Parameters
    ----------
    file_names : list
        The list of file names from the online directory

    Returns
    -------
    filtered_file_names : list
    '''
    return sorted([file_name for file_name in file_names if file_name.endswith('.gz')])


def verify_md5(file_path: str) -> bool:
    '''
    Verify the md5 of a file

    Parameters
    ----------
    file_path : str
        The path of the file to verify
    '''

    with open(file_path, 'rb') as f:
        file_md5 = hashlib.md5(f.read()).hexdigest()

    with open(file_path + '.md5', 'r') as f:
        md5 = f.read().split()[1]

    return file_md5 == md5


def download_verify_uncompress_file(file_url: str, file_path_compressed: str, max_retries: int = 3, backoff_factor: float = 300.0, verbose: bool = True, overwrite: bool = False) -> str | None:
    '''
    Download a file from a url, verify the md5, and uncompress the file

    Parameters
    ----------
    file_url : str
        The url to download the file from
    file_path_compressed : str
        The path to save the file to
    max_retries : int, optional
        The maximum number of retries, by default 3
    backoff_factor : float, optional
        The backoff factor for the exponential backoff, by default 300.0
    verbose : bool
        Whether to print the status of the download
    overwrite : bool, optional
        Whether to overwrite the file if it already exists, by default False

    Returns
    -------
    file_path_uncompressed : str
        The name of the uncompressed file
    '''

    # If any of the folders up to the file path do not exist, create them
    os.makedirs(os.path.dirname(file_path_compressed), exist_ok=True)

    file_path_uncompressed = file_path_compressed[:-3]  # Remove the .gz extension
    file_path_uncompressed_temp = file_path_uncompressed + '.tmp'

    file_path_compressed_md5 = file_path_compressed + '.md5'

    # Remove all temporary files
    for file_path in os.listdir(os.path.dirname(file_path_compressed)):
        if file_path.endswith(os.path.basename(file_path_compressed) + '.tmp'):
            if verbose:
                print(f'Removing temporary file: {file_path}')
            os.remove(os.path.join(os.path.dirname(file_path_compressed), file_path))

    if overwrite:
        # Remove all files
        for file_path in [file_path_uncompressed, file_path_compressed, file_path_compressed_md5]:
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
            logger.error(f'Could not download file: {file_url}')
            return None

        # Verify the md5
        if not verify_md5(file_path_compressed):
            logger.error(f'Could not verify md5 for file: {file_path_compressed}')
            raise Exception(f'Could not verify md5 for file: {file_path_compressed}')

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

    else:
        # Figure out if any of the files already exist and only do the necessary steps

        # Check if the uncompressed file exists
        if os.path.exists(file_path_uncompressed):
            if verbose:
                print(f'File already exists: {file_path_uncompressed}')
                return file_path_uncompressed

        # Download the compressed file
        if not os.path.exists(file_path_compressed):
            try:
                download_file(file_url, file_path_compressed, max_retries=max_retries, backoff_factor=backoff_factor)
            except MaxRetriesExceeded:
                logger.error(f'Could not download file: {file_url}')
                return None

        # Download the md5 file
        if not os.path.exists(file_path_compressed_md5):
            try:
                download_file(file_url + '.md5', file_path_compressed_md5, max_retries=max_retries, backoff_factor=backoff_factor)
            except MaxRetriesExceeded:
                logger.error(f'Could not download file: {file_url}')
                return None

        # Verify the md5
        if not verify_md5(file_path_compressed):
            logger.error(f'Could not verify md5 for file: {file_path_compressed}')
            raise Exception(f'Could not verify md5 for file: {file_path_compressed}')

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

    return file_path_uncompressed


def download_verify_uncompress_file_batch(filtered_file_names: List[str], target_dir: str, n_threads: int = 1, max_retries: int = 3, backoff_factor: float = 300.0, overwrite: bool = False, verbose: bool = True) -> None:
    '''
    Download the files from the online directory

    Parameters
    ----------
    filtered_file_names : list
        The list of file names and their corresponding md5 files
    target_dir : str
        The directory to save the files to
    n_threads : int, optional
        The number of threads to use for downloading, by default 1
    max_retries : int, optional
        The maximum number of retries to attempt, by default 3
    backoff_factor : float, optional
        The backoff factor to use for exponential backoff, by default 300.0
    overwrite : bool
        Whether to overwrite the files if they already exist
    '''
    # Create a progress bar
    pbar = tqdm(total=len(filtered_file_names))
    pbar.set_description('Downloading files')

    files_to_download = filtered_file_names

    with ProcessPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for _ in range(min(n_threads, len(files_to_download))):
            # Get the next file to import
            file_name = files_to_download.pop()

            # Submit the import pipeline to the executor
            futures.append(executor.submit(
                download_verify_uncompress_file,
                file_url=PUBMED_ONLINE_DATA_DIR + file_name,
                file_path_compressed=os.path.join(target_dir, file_name),
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                verbose=verbose,
                overwrite=overwrite))

        while files_to_download:
            logger.debug('Waiting for futures to complete')

            for future in as_completed(futures, timeout=backoff_factor * (2 ** max_retries)):
                # Get the result from the future
                file_path_uncompressed = future.result()

                logger.debug(f'Future {future} result: {file_path_uncompressed}')

                # Check if the download was successful
                if file_path_uncompressed is None:
                    # Download failed, skip the file
                    logger.error(f'Download of file {future} failed.')
                elif not os.path.exists(file_path_uncompressed):
                    # Download successful, but the uncompressed file was not created
                    logger.error(f'Download of file {future} failed. Uncompressed file not created.')
                else:
                    logger.debug(f'Download of file {future} successful.')

                # Remove the future from the list
                futures.remove(future)

                # Submit the next file
                if files_to_download:
                    # Get the next file to import
                    file_name = files_to_download.pop()

                    # Submit the import pipeline to the executor
                    futures.append(executor.submit(
                        download_verify_uncompress_file,
                        file_url=PUBMED_ONLINE_DATA_DIR + file_name,
                        file_path_compressed=os.path.join(target_dir, file_name),
                        max_retries=max_retries,
                        backoff_factor=backoff_factor,
                        verbose=verbose,
                        overwrite=overwrite))

                # Update the progress bar
                pbar.update(1)

    # Wait for all futures to complete
    for future in as_completed(futures, timeout=backoff_factor * (2 ** max_retries)):
        # Get the result from the future
        file_path_uncompressed = future.result()

        logger.debug(f'Future {future} result: {file_path_uncompressed}')

        # Check if the download was successful
        if file_path_uncompressed is None:
            # Download failed, skip the file
            logger.error(f'Download of file {future} failed.')
        elif not os.path.exists(file_path_uncompressed):
            # Download successful, but the uncompressed file was not created
            logger.error(f'Download of file {future} failed. Uncompressed file not created.')
        else:
            logger.debug(f'Download of file {future} successful.')

        # Remove the future from the list
        futures.remove(future)

        # Update the progress bar
        pbar.update(1)

    pbar.set_postfix_str('Done')
    pbar.close()


def download(args: argparse.Namespace) -> None:
    """
    Run the import-download pipeline for a specified number of files.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments to use for the pipeline

    Returns
    -------
    None
    """
    # Display a warning if the number of threads is greater than 2
    if args.n_threads > 2:
        logger.warning(f'Using more than 2 threads (current: {args.n_threads}) may cause problems with the PubMed servers')
        # Also print the warning to stderr
        print(f'WARNING: Using more than 2 threads (current: {args.n_threads}) may cause problems with the PubMed servers', file=sys.stderr)

    # Fetch the files from the online directory
    html = fetch_files(PUBMED_ONLINE_DATA_DIR)
    logger.info('Fetched online directory')

    # Parse the html directory to get all file names
    file_names = parse_html_directory(html)
    logger.info(f'Parsed {len(file_names)} links from the online directory')

    # Filter the file names to only get the .gz filenames
    filtered_file_names = filter_gz_filenames(file_names)
    logger.info(f'Filtered {len(filtered_file_names)} suitable files within the online directory')

    # Download the last files
    if args.last_n_files is not None:
        filtered_file_names = filtered_file_names[-args.last_n_files:]

    # As the user if they want to continue
    if not args.yes:
        answer = input(f'Downloading {len(filtered_file_names)} files into {args.target_dir}. Continue? (y/n) ')
        if answer.lower() not in ['y', 'yes']:
            logger.error('User aborted download')
            raise UserAbort("Confirmation of download")

    # Download the files from the online directory
    logger.info(f'Downloading {len(filtered_file_names)} files into {args.target_dir}')
    download_verify_uncompress_file_batch(
        filtered_file_names,
        target_dir=args.target_dir,
        n_threads=args.n_threads,
        overwrite=args.overwrite,
        max_retries=args.max_retries,
        backoff_factor=args.backoff_factor,
        verbose=args.verbose)
