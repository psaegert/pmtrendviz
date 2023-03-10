def combine_text(doc: dict, include_title: bool = True, include_abstract: bool = True, include_keywords_major: bool = False, include_keywords_minor: bool = False) -> str:
    """
    Combine the text

    Parameters
    ----------
    doc : dict
        The document
    include_title : bool, optional
        Include the title, by default True
    include_abstract : bool, optional
        Include the abstract, by default True
    include_keywords_major : bool, optional
        Include the major keywords, by default False
    include_keywords_minor : bool, optional
        Include the minor keywords, by default False

    Returns
    -------
    str
        The combined text
    """
    # Combine the title, abstract, and keywords into a single string
    doc_title = ''
    if include_title:
        doc_title = doc['_source']['ARTICLE']['ARTICLE_TITLE']

    doc_abstract = ''
    if include_abstract:
        doc_abstract = doc['_source']['ARTICLE']['ABSTRACT']

    doc_keywords_major = ''
    if include_keywords_major:
        doc_keywords_major = ' '.join(doc['_source']['KEYWORDS_MAJOR'])

    doc_keywords_minor = ''
    if include_keywords_minor:
        doc_keywords_minor = ' '.join(doc['_source']['KEYWORDS_MINOR'])

    return ' '.join([doc_title, doc_abstract, doc_keywords_major, doc_keywords_minor])
