"""
Handles text processing, removal and beautifying of HTML and other relevant text operations
"""
import abc
import nltk
import html
import html.parser
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
            html_data = set_has_codeblock(html_data)
        if html_data is None:
            return None
        stripper = HTMLStripper()
        stripper.feed(html_data)
        stripped_html = stripper.get_data()
        # remove newlines from string (since all posts starts/ends with <p>)
        stripped_html = ' '.join(stripped_html.split())
        if clean_dataset:
            stripped_html = set_has_hexadecimal(stripped_html)
            stripped_html = set_has_numeric(stripped_html)
        return stripped_html
    except TypeError as error:
        # print html_data
        print("Error occurred in htmlstripper.strip_tags", error)
    return None


def set_has_codeblock(html_data=str):
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
        print("TypeError in htmlstripper.set_has_codeblock", error)
    return None


def set_has_numeric(text=str):
    """
    Checks the passed text to see if it contains numeric values

    Arguments:
        text (str): Text to remove numeric values from

    Returns:
        str: Processed string (if numeric values), else passed value

    """
    reg_ex = constants.NUMERIC_REG_EX_PATTERN
    if reg_ex.search(text) is None:
        return text
    return reg_ex.sub(constants.QUESTION_HAS_NUMERIC_KEY, text)


def set_has_hexadecimal(text=str):
    """
    Checks the passed text to see if it contains hexadecimal values

    Arguments:
        text (str): Text to remove hexadecimal values from

    Returns:
        str: Processed string (if hexadecimal values), else passed value

    """
    reg_ex = constants.HEXADECIMAL_REG_EX_PATTERN
    if reg_ex.search(text) is None:
        return text
    return reg_ex.sub(constants.QUESTION_HAS_HEXADECIMAL_KEY, text)


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


def set_has_homework():
    """
    Checks if the text contains synonyms to homework, and replaces words with 'has_homework'

    Arguments:


    Returns:


    """

    return


def set_has_tag():
    """
    Checks the text to see if it contains synonyms to homework, and replaces words with 'has_homework'

    Arguments:


    Returns:


    """
    return


def set_has_version_number():
    """

    Arguments:


    Returns:


    """
    return


def set_has_external_link():
    """

    Arguments:


    Returns:


    """
    return



import re
from nltk.corpus import wordnet


test = "this is a long sentence to test if something works as intended. if it doesnt, then there is no point, is there? homeworks homeworks"

t_arr = nltk.word_tokenize(test)

t_dict = {
    "len": int,
    "lem_len": int
}

counter = 0
len_list = list()
word_set = set()
word_len = 0
has_homework = False

hw_list = HOMEWORK_SYNONMS_LIST = [
    "homework",
    "assignment",
    "education",
    "lecture",
    "teach",
    "school",
    "exercise",
    "schoolwork",
    "textbook",
    "college"
    ]

for w in t_arr:
    word_len += len(w)
    w_lem = wordnet.morphy(w, wordnet.NOUN)
    if w_lem is not None and w_lem in hw_list:
        print(w_lem)
        t_dict = {
            "w_len": len(w),
            "w_index": word_len,
            "word": w
        }
        # len_list.append(t_dict)
        len_list.append(w)
        word_set.add(w)
        has_homework = True
    counter += 1

print(t_arr)
print(len_list)
print(list(set(len_list)))
print(word_set)

for value in word_set:  # len_list:
    # w_len = value.get("w_len")
    # w_index = value.get("w_index")
    # word = value.get("word")
    print(test.replace(value, "TEST"))
#
# for word in t_arr:
#     if (wordnet.morphy(word, wordnet.NOUN)) is not None:
#         print(wordnet.morphy(word, wordnet.NOUN))
#     # print(word+"-->"+WordNetLemmatizer().lemmatize(word,'v'))
#
# # print(test.replace()[6])
#
# print(wordnet.morphy(test, wordnet.NOUN))
#
# print([(m.start(), m.span()) for m in re.finditer('if', test)])

