import abc
import html
import html.parser

from bs4 import BeautifulSoup
from html.parser import HTMLParser

from constants import QUESTION_HAS_CODE_KEY


class HTMLStripper(HTMLParser):
    """
    Removes HTML elements from the received text.

    Source: http://stackoverflow.com/questions/753052/strip-html-from-strings-in-python

    Accepted answer: "answered May 29 '09 at 11:47 Eloff".

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


def strip_tags(html_data):
    """
    Returns a string without HTML elements

    Arguments:
        html_data (str): HTML text to convert to string

    Returns:
        str: String without HTML elements || None (if error)

    """
    try:
        html_data = html.unescape(html_data)
        stripper = HTMLStripper()
        html_data = set_has_codeblock(html_data)
        if html_data is None:
            return None
        stripper.feed(html_data)
        return stripper.get_data()
    except TypeError as error:
        # print html_data
        print("Error occurred in htmlstripper.strip_tags", error)
    return None


def set_has_codeblock(html_data=str, encoding='utf-8'):
    """
    Replaces the content of the <code> tag (if exists) with the value 'has_codeblock'

    Questions can be of both different length and contain more than one question. In
    addition, the question can have one or more code examples added to it ```<code>```.
    In this function, BeautifulSoup and ```BeautifulSoup.find_all()``` is used to replace
    the content of all <code> elements. This way, instead of having a large code sample,
    you only get one word/term; ```has_codeblock```.

    Arguments:
        html_data (str): The HTML text to search and replace <code> text
        encoding (str, default='utf-8'): Encoding for the HTML string

    Returns:
        str: html_data with where <code> value has been replaced by 'has_codeblock'

    See:
    ```constants.QUESTION_HAS_CODE_KEY```
    """
    try:
        find = "code"
        bsoup = BeautifulSoup(html_data, "html.parser")
        for child in bsoup.find_all(find):
            child.string = QUESTION_HAS_CODE_KEY
        return bsoup.prettify()
    except TypeError as error:
        print("TypeError in htmlstripper.set_has_codeblock", error)
    return None
