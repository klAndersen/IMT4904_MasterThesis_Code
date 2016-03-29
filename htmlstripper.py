from lxml import etree
from HTMLParser import HTMLParser
from bs4 import BeautifulSoup


class HTMLStripper(HTMLParser):
    """
    Removes HTML elements from the received text.

    Source: http://stackoverflow.com/questions/753052/strip-html-from-strings-in-python

    Accepted answer: "answered May 29 '09 at 11:47 Eloff".

    And comment to this answer by 'pihentagyu aka James Doepp' (May 21 '15 at 17:49)
    """

    def __init__(self):
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    """
    Returns a string without HTML elements

    Arguments:
        html (str): HTML text to convert to string

    Returns:
        str: String without HTML elements

    """
    try:
        html = HTMLParser().unescape(html)
        stripper = HTMLStripper()
        html = remove_code_element_from_html(html)
        stripper.feed(html)
        return stripper.get_data()
    except TypeError as error:
        print "=============ERROR========="
        print html
        print error.message
    return None


def remove_code_element_from_html(html=str, encoding='utf-8'):
    """
    Removes the code block from the text by looking for the <code> tag

    Since Questions can contain varying size of code examples, this needs to be
    removed for improved feature extraction. This function looks for code
    samples by looking for the <code> element. When found, it is removed, but the
    tail (if any) is added to the parent to which the <code> element belonged.

    The tail is the text (or content) following the given <code> element
    (which would be removed if not kept). After all <code> elements has been
    removed, the cleaned html string is returned.

    To also take in account bad HTML/invalid HTML tags, BeatifulSoup is used
    to add end tags for those missing it.

    Arguments:
        html (str): The HTML text to remove <code> blocks from
        encoding (str, default='utf-8'): Encoding for the HTML string

    Returns:
        str: String where the <code> tag and its content has been removed
    """
    try:
        find = "code"
        tail_attr = "tail"
        # account for bad html tags
        bsoup = BeautifulSoup(html, 'lxml-xml')
        html = bsoup.prettify(encoding)
        root = etree.fromstring(html)
        for child in root.iter(find):
            parent = child.getparent()
            parent.remove(child)
            print "attr", getattr(parent, tail_attr, None)
            print "attr", getattr(child, tail_attr, None)
            # to avoid loss of tail, add it to parent
            if getattr(child, tail_attr, None) is not None:
                if getattr(parent, tail_attr, None) is None:
                    parent.tail = ""
                parent.tail += getattr(child, tail_attr, '')
        return etree.tostring(root)
    except etree.XMLSyntaxError as error:
        # print html
        print error.message
    return None


html_t = "<p> a simple <code> test </code> </p>"
print remove_code_element_from_html(html_t)
