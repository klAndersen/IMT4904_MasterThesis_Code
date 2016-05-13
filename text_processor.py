"""
Handles text processing, removal and beautifying of HTML and other relevant text operations
"""
import abc
import nltk
import html
import html.parser
from nltk.corpus import wordnet
from html.parser import HTMLParser

from bs4 import BeautifulSoup

import constants


class HTMLStripper(HTMLParser):
    """
    Removes HTML elements from the received text.

    Source: http://stackoverflow.com/questions/753052/strip-html-from-strings-in-python

    Accepted answer (http://stackoverflow.com/a/925630): "answered May 29 '09 at 11:47 Eloff".

    And comment to this answer by 'pihentagyu aka James Doepp' (May 21 '15 at 17:49)
    """

    __metadata__ = abc.ABCMeta

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)

    @abc.abstractmethod
    def handle_entityref(self, name):
        pass

    @abc.abstractmethod
    def handle_charref(self, name):
        pass


def stem_training_data(stemming_data=str):
    """
    Removes affixes and returns the stem

    Arguments:
        stemming_data (str): Data to stem

    Returns:
        str: String containing the stemmed data.
        E.g. the words 'cry, 'crying', 'cried' would all return 'cry'.

    """
    porter = nltk.PorterStemmer()
    stemming_data = stemming_data.lower().split()
    stemming_data = map(lambda x: porter.stem(x), stemming_data)
    return ' '.join(stemming_data)


def strip_tags(html_data, clean_dataset=True):
    """
    Returns a string without HTML elements and newlines

    Arguments:
        html_data (str): HTML text to convert to string
        clean_dataset (bool): Should questions be cleaned (e.g. remove code samples, hexadecimals, numbers, etc)?

    Returns:
        str: String without HTML elements || None (if error)

    """
    try:
        html_data = html.unescape(html_data)
        if clean_dataset:
            html_data = __set_has_codeblock(html_data)
        if html_data is None:
            return None
        stripper = HTMLStripper()
        stripper.feed(html_data)
        stripped_html = stripper.get_data()
        # remove newlines from string (since all posts starts/ends with <p>)
        stripped_html = ' '.join(stripped_html.split())
        if clean_dataset:
            stripped_html = __set_has_hexadecimal(stripped_html)
            stripped_html = __set_has_numeric(stripped_html)
        return stripped_html
    except TypeError as error:
        # print html_data
        print("Error occurred in text_processor.strip_tags", error)
    return None


def __set_has_codeblock(html_data=str):
    """
    Replaces the content of the <code> tag (if exists) with the value 'has_codeblock'

    Questions can be of both different length and contain more than one question. In
    addition, the question can have one or more code examples added to it ```<code>```.
    In this function, BeautifulSoup and ```BeautifulSoup.find_all()``` is used to replace
    the content of all <code> elements. This way, instead of having a large code sample,
    you only get one word/term; ```has_codeblock```.

    Arguments:
        html_data (str): The HTML text to search and replace <code> text

    Returns:
        str: Returns the processed ```html_data```

    See:
    ```constants.QUESTION_HAS_CODE_KEY```
    """
    try:
        find = "code"
        bsoup = BeautifulSoup(html_data, "html.parser")
        for child in bsoup.find_all(find):
            child.string = constants.QUESTION_HAS_CODEBLOCK_KEY
        return bsoup.prettify()
    except TypeError as error:
        print("TypeError in text_processor.__set_has_codeblock", error)
    return None


def __set_has_numeric(text=str):
    """
    Checks the passed text to see if it contains numeric values

    Arguments:
        text (str): Text to check for numeric values

    Returns:
        str: Processed string (if numeric values), or the original text

    """
    reg_ex = constants.NUMERIC_REG_EX_PATTERN
    if reg_ex.search(text) is None:
        return text
    return reg_ex.sub(constants.QUESTION_HAS_NUMERIC_KEY, text)


def __set_has_hexadecimal(text=str):
    """
    Checks the passed text to see if it contains hexadecimal values

    Arguments:
        text (str): Text to check for hexadecimal values

    Returns:
        str: Processed string (if hexadecimal values), or the original text

    """
    reg_ex = constants.HEXADECIMAL_REG_EX_PATTERN
    if reg_ex.search(text) is None:
        return text
    return reg_ex.sub(constants.QUESTION_HAS_HEXADECIMAL_KEY, text)


def __set_has_homework(text=str):
    """
    Checks if the text contains synonyms to homework, and replaces words with 'has_homework'

    Arguments:
        text (str): Text to check for "homework words"

    Returns:
        str: Text with replaced "homework words", or the original text

    """
    word_set = set()
    has_homework_key = constants.QUESTION_HAS_HOMEWORK_KEY
    homework_words = constants.HOMEWORK_SYNONMS_LIST
    tokenized_text = nltk.word_tokenize(text)
    # loop through all the words to see if it contains homework or its synonyms
    for word in tokenized_text:
        word_lem = wordnet.morphy(word, wordnet.NOUN)
        if (word_lem is not None) and (word_lem in homework_words):
            word_set.add(word)
    # replace those words, if any, with 'has_homework'
    for word in word_set:
        text = text.replace(word, has_homework_key)
    return text


def __find_and_replace_words(text=str, word_list=list, replacement_text=str):
    """
    Checks if the passed text contains any of the words in the passed list

    Arguments:
        text (str): Text to match against words in list
        word_list (list): List to use as comparison against the words in the text
        replacement_text (str): Text to use for word replacement

    Returns:
        str: Text with replaced words, or the original text

    """
    word_set = set()
    tokenized_text = nltk.word_tokenize(text)
    # loop through all the words to see if it contains any of the words in the list
    for word in tokenized_text:
        if word in word_list:
            word_set.add(word)
    # replace those words, if any, with the replacement words
    for word in word_set:
        text = text.replace(word, replacement_text)
    return text


def __set_has_tag(text=str, text_tags=list, site_tags=list):
    """
    Checks the text to see if it contains any of the tags found on StackOverflow, and replaces them

    Arguments:
        text (str): Text to check for (and replace) tag values
        text_tags (list): Tags attached to the question
        site_tags (list): Tags found on the site (here: StackOverflow)

    Returns:
        str: Text without tags (where all tags have been replaced with 'has_*_tag), or the original text

    """
    has_attached_tag_key = constants.QUESTION_HAS_ATTACHED_TAG_KEY
    has_external_tag_key = constants.QUESTION_HAS_EXTERNAL_TAG_KEY
    updated_text = __find_and_replace_words(text, text_tags, has_attached_tag_key)
    updated_text = __find_and_replace_words(updated_text, site_tags, has_external_tag_key)
    return updated_text


def __set_has_version_number(text=str):
    """
    Checks if the text contains any forms of version numbering.
    If it does, these are replaced with 'has_version_number'

    Arguments:
        text (str): Text to check for (and replace) version number

    Returns:
        str: Text without version numbering, or the original text

    """
    has_version_number = constants.QUESTION_HAS_VERSION_NUMBER_KEY

    return text


def __set_has_link(html_text=str):
    """
    Checks if the text contains any links. If so, these are replaced by 'has_links'

    Arguments:
        html_text (str): Text to check for (and replace) links

    Returns:
        str: Text without links, or the original text

    """
    try:
        find = "a"
        bsoup = BeautifulSoup(html_text, "html.parser")
        for child in bsoup.find_all(find):
            child.string = constants.QUESTION_HAS_LINKS_KEY
        return bsoup.prettify()
    except TypeError as error:
        print("TypeError in text_processor.__set_has_link", error)
    return None
