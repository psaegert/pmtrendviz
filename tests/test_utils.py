import os
import shutil

import pandas as pd
import pytest

from pmtrendviz.utils import ImportOptimizer, MaxRetriesExceeded, UserAbort, download_file, \
    extract_dates_from_document, get_articles_per_date, get_cache_dir, get_data_dir, \
    get_es_client, get_models_dir, get_time_to_complete_offset


def test_get_es_client() -> None:
    """Test the get_es_client function."""
    # Test with default arguments
    es = get_es_client()
    assert es is not None
    assert es.ping()


def test_user_abort() -> None:
    """Test the UserAbort exception."""
    try:
        raise UserAbort('test')
    except UserAbort as e:
        assert str(e) == 'User aborted the program at: test.'


@pytest.mark.usefixtures('manage_pytest_folder')
def test_download_file(manage_pytest_folder: str) -> None:
    """Test the download_file function."""
    try:
        download_file('https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/README.txt', os.path.join(manage_pytest_folder, 'pytest_README.txt'))
        assert os.path.exists(os.path.join(manage_pytest_folder, 'pytest_README.txt'))
    except MaxRetriesExceeded:
        pass

    if os.path.exists(os.path.join(manage_pytest_folder, 'pytest_README.txt')):
        os.remove(os.path.join(manage_pytest_folder, 'pytest_README.txt'))


def test_extract_dates_from_document() -> None:
    """Test the extract_dates_from_document function."""
    doc = {'_source': {'ARTICLE': {'ARTICLE_DATE': '2020-01-01'}, 'DATE_COMPLETED': '2020-01-02', 'DATE_REVISED': '2020-01-03'}}

    dates = extract_dates_from_document(doc)
    assert dates == {'article_date': '2020-01-01', 'date_completed': '2020-01-02', 'date_revised': '2020-01-03'}


def test_extract_dates_from_document_parse() -> None:
    """Test the extract_dates_from_document function."""
    doc = {'_source': {'ARTICLE': {'ARTICLE_DATE': '2020-01-01'}, 'DATE_COMPLETED': '2020-01-02', 'DATE_REVISED': '2020-01-03'}}

    dates = extract_dates_from_document(doc, parse_dates=True)

    assert dates == {'article_date': pd.Timestamp('2020-01-01'), 'date_completed': pd.Timestamp('2020-01-02'), 'date_revised': pd.Timestamp('2020-01-03')}


class TestImportOptimizer:
    index = 'pytest_import_optimizer_index'
    optimizer = ImportOptimizer(index)

    def setup_method(self) -> None:
        # Clear the files from the previous test
        shutil.rmtree(os.path.join(get_data_dir(), self.index), ignore_errors=True)

    def teardown_method(self) -> None:
        # Clear the files from the previous test
        shutil.rmtree(os.path.join(get_data_dir(), self.index), ignore_errors=True)

    def test_register_imported_file(self) -> None:
        self.optimizer.register_imported_file('test_file')

        index_file = os.path.join(get_data_dir(), "imported_files", f"{self.index}.txt")

        assert os.path.exists(index_file)

        with open(index_file, 'r') as f:
            assert f.read() == 'test_file\n'

    def test_filter_registered_files(self) -> None:
        self.optimizer.register_imported_file('test_file')

        assert self.optimizer.filter_registered_files(['test_file', 'test_file2']) == ['test_file2']

    def test_reset_imported_files(self) -> None:
        self.optimizer.register_imported_file('test_file')

        self.optimizer.reset_imported_files()

        index_file = os.path.join(get_data_dir(), "imported_files", f"{self.index}.txt")

        assert not os.path.exists(index_file)


def test_get_models_dir() -> None:
    """Test the get_models_dir function."""
    models_dir = get_models_dir()
    assert os.path.exists(models_dir)


def test_get_cache_dir() -> None:
    """Test the get_cache_dir function."""
    cache_dir = get_cache_dir()
    assert os.path.exists(cache_dir)


@pytest.mark.usefixtures('manage_pytest_index')
def test_get_time_to_complete_offset() -> None:
    """Test the get_time_to_complete_offset function."""
    es = get_es_client()
    # Print the mapping
    print(es.indices.get_mapping(index='pytest'))
    offset = get_time_to_complete_offset(index='pytest')
    assert type(offset) == int


@pytest.mark.usefixtures('manage_pytest_index')
def test_get_articles_per_date() -> None:
    """Test the get_articles_per_date function."""
    articles_per_date = get_articles_per_date(index='pytest')
    assert articles_per_date.columns.tolist() == ['date', 'articles', 'rank']
    assert articles_per_date['date'].dtype == 'datetime64[ns]'
    assert articles_per_date.sort_values('articles', ascending=False)['rank'].is_monotonic_increasing
