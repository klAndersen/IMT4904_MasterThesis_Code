"""
Source: http://stackoverflow.com/questions/753052/strip-html-from-strings-in-python

Accepted answer: "answered May 29 '09 at 11:47 Eloff"
And comment to this answer by 'pihentagyu aka James Doepp' (May 21 '15 at 17:49)
"""

from HTMLParser import HTMLParser


class HTMLStripper(HTMLParser):

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
    html = HTMLParser().unescape(html)
    stripper = HTMLStripper()
    stripper.feed(html)
    return stripper.get_data()
