from re import compile, IGNORECASE, VERBOSE

"""
File containing constants used in more than one place
"""
QUESTION_VOTES_KEY = "Score"
'''
Column identifier/key: Score

Amount of votes/scores on a given question
'''

QUESTION_TITLE_KEY = "Title"
'''
Column identifier/key: Title

The title of the question as seen on StackOverflow
'''

QUESTION_TEXT_KEY = "Body"
'''
Column identifier/key: Body

The question text
'''

CLASS_LABEL_KEY = "label"
'''
Column identifier/key: label

Label for the retrieved training data.
'''

QUESTION_LENGTH_KEY = "length"
'''
Column identifier/key: length

The length of the questions text
'''

QUESTION_HAS_CODEBLOCK_KEY = " has_codeblock "
'''
Value used to replace the code sample found in the <code> tags.
To avoid issues when concatenating strings, space is added before and after.
'''

QUESTION_HAS_HEXADECIMAL_KEY = " has_hexadecimal "
'''
Value used to replace hexadecimal values in the question text.
To avoid issues when concatenating strings, space is added before and after.
'''

QUESTION_HAS_NUMERIC_KEY = " has_numeric "
'''
Value used to replace numeric values in the question text.
To avoid issues when concatenating strings, space is added before and after.
'''

NUMERIC_REG_EX_PATTERN = compile(r'[+-]?\b\d+\b')
'''
Regular expression to check for numeric values
'''

HEXADECIMAL_REG_EX_PATTERN = compile(r"""
    (\b[^0-9a-f]0x\b)       # group 1: look only for the exact hex value '0x'
    |
    ([^0-9a-f]0x[0-9A-Z]+)  # group 2: look for hex values starting with '0x'
    """, IGNORECASE + VERBOSE)
'''
Regular expression to check for hexadecimal values
'''
