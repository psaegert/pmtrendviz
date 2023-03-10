import argparse
import datetime
import json
import logging
import os
import signal
import xml.etree.ElementTree as ET
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from glob import glob
from typing import Dict, List, Set, Tuple, Union

from tqdm import tqdm

from ..utils import UserAbort

logger = logging.getLogger('pmtrendviz.import_pipeline.extract')


def parse_medline_citation(elem: ET.Element) -> Dict[str, Union[str, datetime.date | None, List[str], int, Dict]]:
    """
    Parse one Medline citation element (holds the article information) of XML database to python dictionary

    Parameters
    ----------
    elem: ET.Element

    Returns
    -------
    medline citation object: dict
    """
    mc: Dict[str, Union[str, datetime.date | None, List[str], int, Dict]] = dict()
    pmid_elem = elem.find('PMID')
    if pmid_elem is not None and pmid_elem.text is not None:
        mc['PMID'] = int(pmid_elem.text)
    else:
        logger.error('PMID not found')
        raise ValueError('PMID not found')

    mc['DATE_REVISED'] = parse_date(elem.find('DateRevised'))
    mc['DATE_COMPLETED'] = parse_date(elem.find('DateCompleted'))
    mc['KEYWORDS_MAJOR'], mc['KEYWORDS_MINOR'] = parse_keywords(elem.find('KeywordList'))
    mc['ARTICLE'] = parse_article(elem.find('Article'))
    return mc


def parse_article(elem: ET.Element | None) -> Dict[str, Union[str, datetime.date | None, List[dict]]]:
    """
    parse one article instance of the XML database to a python dictionary
    Parameters
    ----------
    elem: ET.Element

    Returns
    -------
    article object: dict
    """
    a: Dict[str, Union[str, datetime.date | None, List[dict]]] = dict()
    if elem is not None:
        title_elem = elem.find('ArticleTitle')
        if title_elem is not None:
            a['ARTICLE_TITLE'] = ET.tostring(title_elem, encoding='unicode', method='text')
        else:
            a['ARTICLE_TITLE'] = ''
        texts = []
        abstract_elem = elem.find('Abstract')  # localize abstract
        if abstract_elem:  # if abstract is available in PubMed
            for child in abstract_elem.iter('AbstractText'):  # iterate over all parts of the abstract (abstract may be
                # chopped up into multiple parts reflecting the inner structure of the article)
                texts.append(ET.tostring(child, encoding='unicode', method='text'))
        a['ABSTRACT'] = ' '.join(filter(lambda x: x is not None, texts))  # join parts to create one abstract text
        language_elem = elem.find('Language')
        if language_elem is not None and language_elem.text is not None:
            a['LANGUAGE'] = language_elem.text
        else:
            a['LANGUAGE'] = ''
        a['ARTICLE_DATE'] = parse_date(elem.find('ArticleDate'))
        authors_elem = elem.find('AuthorList')
        if authors_elem is not None:
            a['AUTHORS'] = [parse_author(e) for e in authors_elem.iter('Author')]  # multiple authors possible -> iterate over all "Author" fields
        else:
            a['AUTHORS'] = []
    return a


def parse_author(elem: ET.Element | None) -> Dict[str, str]:
    """
    Parses one author element found in the XML database into an python dictionary.
    Either author is given by first_name, last_name or a collective_name (e.g. a group's/ institute's name).

    Parameters
    ----------
    elem: ET.Element

    Returns
    -------
    author_object: dict
    """
    a: Dict[str, str] = dict()
    if elem is not None:
        last_name_elem = elem.find('LastName')
        a['LAST_NAME'] = ''
        if last_name_elem is not None and last_name_elem.text is not None:
            a['LAST_NAME'] = last_name_elem.text
        first_name_elem = elem.find('FIRST_NAME')
        a['FIRST_NAME'] = ''
        if first_name_elem is not None and first_name_elem.text is not None:
            a['FIRST_NAME'] = first_name_elem.text
        collective_name_elem = elem.find('COLLECTIVE_NAME')
        a['COLLECTIVE_NAME'] = ''
        if collective_name_elem is not None and collective_name_elem.text is not None:
            a['COLLECTIVE_NAME'] = collective_name_elem.text
    return a


def parse_keywords(elem: ET.Element | None) -> Tuple[List[str], List[str]]:
    """
    Parses keywords of the XML database to lists.
    Keywords are given as XML elements with attributes specifying if they are major or minor keywords.
    The function collects them to two lists, respectively.

    Parameters
    ----------
    elem: ET.Element

    Returns
    -------
    list of major and minor keywords as strings respectively
    """
    keywords_major = []
    keywords_minor = []
    if elem is not None:
        for child in elem:
            major_topic = child.attrib['MajorTopicYN']
            if child.text is not None:
                if major_topic == 'Y':  # keyword is major
                    keywords_major.append(child.text)
                else:  # keyword is minor
                    keywords_minor.append(child.text)
    return keywords_major, keywords_minor


def parse_date(elem: ET.Element | None) -> datetime.date | None:
    """
    Function parses dates from XML database to python's datetime.date.
    Since dates are split up in YEAR, MONTH, DAY attributes int the XML, transform them to a python data structure,
    which can be outputted in a format to be read by ElasticSearch.

    Parameters
    ----------
    elem: ET.Element

    Returns
    -------
    date: datetime.date
    """
    # TODO: Add handling of different date formats
    if elem is None:
        return None

    for child in elem:
        key = child.tag
        if key == 'Year' and child.text is not None:
            year = int(child.text)
        elif key == 'Month' and child.text is not None:
            month = int(child.text)
        elif key == 'Day' and child.text is not None:
            day = int(child.text)
    return datetime.date(year, month, day)


def parse_xml(xml_file_path: str) -> List[dict]:
    """
    Function to parse the articles found in each XML file to a list of dictionaries representing the relevant XML
    fields.

    Parameters
    ----------
    xml_file_path: str

    Returns
    -------
    articles: list[dict]
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    articles = []
    if root is not None:
        i: int = 0
        for xml_elem in root.findall('PubmedArticle/MedlineCitation'):  # Find the elements, in which information about
            # the articles are stored
            article_dict = parse_medline_citation(xml_elem)
            articles.append(article_dict)
            i += 1
        logger.info(f'Parsed {i} articles from {xml_file_path}')
    else:
        logger.error(f'XML file {xml_file_path} is empty')
    return articles


def extract_worker(xml_file_path: str, json_file_path: str) -> None:
    """
    Function, which parses the given XML file and stores the relevant data of the XML to a JSON file, which can be read
    by ElasticSearch (including the relevant metadata).

    Parameters
    ----------
    xml_file_path: str
    json_file_path: str

    Returns
    -------

    """
    articles = parse_xml(xml_file_path)
    logger.debug(f'Writing {len(articles)} articles to {json_file_path}')
    with open(json_file_path, 'a') as f:
        for article in articles:
            f.write(f'{{"index": {{"_id": {article["PMID"]}}}}}\n')
            f.write(json.dumps(article, default=str) + '\n')


def extract(args: argparse.Namespace) -> None:
    """
    Run the extract pipeline for all files in the source directory and store the extracted data in json files in the target directory.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the program

    Returns
    -------
    None
    """

    xml_file_names: List[str] = sorted([file_name for file_name in glob(os.path.join(args.src_dir, '*.xml'))])
    json_file_names: List[str] = [
        os.path.join(args.target_dir, os.path.splitext(os.path.split(xml_file_name)[1])[0] + '.json') for xml_file_name in
        xml_file_names]

    # As the user if they want to continue
    if not args.yes:
        answer = input(f'Extracting data from {len(xml_file_names)} files into {args.target_dir}. Continue? (y/n) ')
        if answer.lower() not in ['y', 'yes']:
            logger.error('User aborted extraction')
            raise UserAbort("Confirmation of extraction")

    if args.overwrite:
        for json_file_path in json_file_names:
            try:
                os.remove(json_file_path)
            except FileNotFoundError:
                pass
    else:
        # Ignore files which already exist
        xml_file_names = [xml_file_name for xml_file_name, json_file_name in zip(xml_file_names, json_file_names) if not os.path.exists(json_file_name)]
        json_file_names = [json_file_name for json_file_name in json_file_names if not os.path.exists(json_file_name)]

    pbar = tqdm(total=len(xml_file_names))

    # use a process pool to extract the data from the files
    with ProcessPoolExecutor(args.n_threads) as executor:
        fts: dict = dict()
        try:
            for xml_file_path, json_file_path in zip(xml_file_names, json_file_names):  # iterate over all XML files
                pbar.set_description(f'Processing {xml_file_path}', refresh=True)
                # enqueue extracting task per file
                ft: Future = executor.submit(extract_worker, xml_file_path, json_file_path)
                fts[ft] = xml_file_path

            extracted_futures: Set[Future] = set()
            while len(extracted_futures) < len(xml_file_names):
                # get all results of finished tasks in done
                done, not_done = wait(fts, timeout=1, return_when=FIRST_COMPLETED)

                new_futures = done - extracted_futures
                # Handle the first new future
                if len(new_futures) > 0:
                    future = new_futures.pop()
                    extracted_futures.add(future)
                    logger.debug(f'Extracted data from {fts[future]} and saved to JSON file')

        except KeyboardInterrupt:
            logger.error('Keyboard interrupt during extraction')
            for pid in executor._processes:  # HACK: kill all processes of executor (may break in future versions)
                os.kill(pid, signal.SIGTERM)

        pbar.set_postfix_str('Done')
        pbar.close()
