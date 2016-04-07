import abc
import html
import html.parser
from lxml import etree
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
        # print html
        print("Error occurred in htmlstripper.strip_tags", error)
    return None


def set_has_codeblock(html_data=str, encoding='utf-8'):
    """
    Replaces the content of the <code> tag (if exists) with the value 'has_codeblock'

    Since Questions can contain varying size of code examples, the content can create
    a lot of bad features. To account for this, the code examples are replaced with a
    single value indicating that this question contains one or more code examples.
    This is done by converting the HTML into XML and looking for the <code> tag.

    If the question contains the <code> tag, the value (the text) is replaced by
    a 'has_codeblock' value. To account for bad HTML/invalid HTML tags, BeatifulSoup
    is used to add end tags for those missing it.

    Arguments:
        html_data (str): The HTML text to search and replace <code> text
        encoding (str, default='utf-8'): Encoding for the HTML string

    Returns:
        str: HTML string where the <code> tag contains a 'has_codeblock' value

    See:
    ```constants.QUESTION_HAS_CODE_KEY```
    """
    try:
        find = "code"
        # account for bad html tags
        bsoup = BeautifulSoup(html_data, 'lxml-xml')
        html_data = bsoup.prettify(encoding)
        root = etree.fromstring(html_data)
        for child in root.iter(find):
            child.text = QUESTION_HAS_CODE_KEY
        return etree.tostring(root, encoding="unicode")
    except TypeError as error:
        print("TypeError in htmlstripper.set_has_codeblock", error)
    except etree.XMLSyntaxError as error:
        # print html
        print("XMLSyntaxError in htmlstripper.set_has_codeblock", error)
    return None

