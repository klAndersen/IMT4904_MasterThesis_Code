import abc
import html
import html.parser
from lxml import etree
from bs4 import BeautifulSoup
from html.parser import HTMLParser


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
        str: String without HTML elements || None upon failure

    """
    try:
        html_data = html.unescape(html_data)
        stripper = HTMLStripper()
        html_data = remove_code_element_from_html(html_data)
        if html_data is None:
            return None
        stripper.feed(html_data)
        return stripper.get_data()
    except TypeError as error:
        # print html
        print("Error occurred in htmlstripper.strip_tags", error)
    return None


def remove_code_element_from_html(html_data=str, encoding='utf-8'):
    """
    Removes the code block from the text by looking for the <code> tag

    Since Questions can contain varying size of code examples, this needs to be
    removed for improved feature extraction. This function looks for code
    samples by looking for the <code> element. When found, it is removed, but the
    tail^ (if any) is added to the parent to which the <code> element belonged.

    To also take in account bad HTML/invalid HTML tags, BeatifulSoup is used
    to add end tags for those missing it.

    ^The tail is the text (or content) following the given <code> element
    (which would be removed if not kept). After all <code> elements has been
    removed, the cleaned html string is returned.

    Arguments:
        html_data (str): The HTML text to remove <code> blocks from
        encoding (str, default='utf-8'): Encoding for the HTML string

    Returns:
        str: String where the <code> tag and its content has been removed
    """
    try:
        find = "code"
        tail_attr = "tail"
        # account for bad html tags
        bsoup = BeautifulSoup(html_data, 'lxml-xml')
        html_data = bsoup.prettify(encoding)
        root = etree.fromstring(html_data)
        for child in root.iter(find):
            parent = child.getparent()
            parent.remove(child)
            # to avoid loss of tail, add it to parent
            if getattr(child, tail_attr, None) is not None:
                if getattr(parent, tail_attr, None) is None:
                    parent.tail = ""
                parent.tail += getattr(child, tail_attr, '')
        return etree.tostring(root, encoding="unicode")
    except TypeError as error:
        print("TypeError occurred in htmlstripper.remove_code_element_from_html", error)
    except etree.XMLSyntaxError as error:
        # print html
        print("XMLSyntaxError occurred in htmlstripper.remove_code_element_from_html", error)
    return None

