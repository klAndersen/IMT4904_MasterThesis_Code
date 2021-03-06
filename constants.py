"""
File containing various constants used in this project
"""

import multiprocessing
from sys import platform
from re import compile, IGNORECASE, VERBOSE

"""
    https://docs.python.org/3.5/library/sys.html#sys.platform

    |   ------------------------------
    |   System      |   platform value
    |   ------------------------------
    |   Linux       |   'linux'
    |   Windows     |   'win32'
    |   Windows/Cygwin   |   'cygwin'
    |   Mac OS X    |   'darwin'
"""
PLATFORM_IS_WINDOWS = platform.startswith("win32")
"""
Constant value for OS Platform: Windows (win32)
"""

PLATFORM_IS_CYGWIN = platform.startswith("cygwin")
"""
Constant value for OS Platform: CygWin
"""

PLATFORM_IS_LINUX = platform.startswith("linux")
"""
Constant value for OS Platform: Linux
"""

PLATFORM_IS_MAC = platform.startswith("darwin")
"""
Constant value for OS Platform: Mac (darwin)
"""

WINDOWS_PATH_SEPARATOR = "\\"
"""
Path separator for Windows systems, e.g:
C:\\Users\\<username>\\Documents
"""

LINUX_PATH_SEPARATOR = "/"
"""
Path separator for Linux and Unix, e.g:
/home/<username>/Documents/
"""


def get_platform_separator():
    """
    Returns the correct path separator for the OS running this program (e.g. Windows: '\\', *nix: '/'

    Returns:
         str: The path separator for the OS running this program
    """
    if PLATFORM_IS_WINDOWS:
        sign = WINDOWS_PATH_SEPARATOR
    else:
        sign = LINUX_PATH_SEPARATOR
    return sign


DATABASE_LIMIT = {
    '10': 10,
    '100': 100,
    '1000': 1000,
    '10000': 10000
}

SEPARATOR = get_platform_separator()

FILEPATH_FEATURE_DETECTOR = "." + SEPARATOR + "feature_detectors" + SEPARATOR
"""
The path to where the different feature detectors can be found
"""

FILEPATH_TRAINING_DATA = "." + SEPARATOR + "training_data" + SEPARATOR
"""
The path to where the training data can be found
"""

FILEPATH_MODELS = "." + SEPARATOR + "pickle_models" + SEPARATOR
"""
Path to folder containing the produced classifier models
"""

FILEPATH_SINGULAR_FEATURE_MODELS = "." + SEPARATOR + "feat_detect_models" + SEPARATOR
"""
Path to folder containing the produced classifier models based on a singular feature detector
"""

TAG_NAME_COLUMN = "TagName"
"""
The column name of the Tags in the database (from the table Tags).
Also the column name used in pandas.DataFrame for these Tags.
DO NOT CHANGE THIS VALUE!
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

SET_UNPROCESSED = "_unprocessed"
'''
Default value attached to the raw, unprocessed data sets

Can also be used to check if the loaded data is unprocessed, if so remove HTML
'''

QUESTION_HAS_ATTACHED_TAG_KEY = " has_attached_tag "
'''
Value used to replace the tag value, if found.
Since questions come with attached tags, if these are mentioned in question,
these are replaced with the ```has_attached_tag```.
To avoid issues when concatenating strings, space is added before and after.
'''

QUESTION_HAS_EXTERNAL_TAG_KEY = " has_external_tag "
'''
Value used to replace the tag value, if found.
Some questions can come with tags which aren't attached to the given question.
If that's the case, these values are replaced with the ```has_external_tag```.
To avoid issues when concatenating strings, space is added before and after.
'''

QUESTION_HAS_HOMEWORK_KEY = " has_homework "
'''
If the text contains any words that are synonyms to homework, replace them with this value.
To avoid issues when concatenating strings, space is added before and after.
'''

QUESTION_HAS_ASSIGNMENT_KEY = " has_assignment "
'''
If the text contains the word 'assignment', replace it.
To avoid issues when concatenating strings, space is added before and after.
'''

QUESTION_HAS_VERSION_NUMBER_KEY = " has_version_number "
'''
If the question contains version numbering, this value replaces it.
The values to replace should be both the version text (e.g. 'v.', 'vno', 'version', etc),
and the numeric value that follows.
To avoid issues when concatenating strings, space is added before and after.
'''

QUESTION_HAS_LINKS_KEY = " has_links "
'''
Value used to replace links found in the text
To avoid issues when concatenating strings, space is added before and after.
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

HOMEWORK_SYNONMS_LIST = [
    "homework",
    "education",
    "lecture",
    "teach",
    "teacher",
    "supervisor",
    "professor",
    "schoolteacher",
    "school",
    "exercise",
    "schoolwork",
    "textbook",
    "exam",
    "college"
]
"""
Array containing synonyms for the word 'homework'.
Source: http://www.thesaurus.com/browse/homework
"""

ASSIGNMENT_LIST = [
    "assignment",
]
"""
Array containing only the word assignment.
Originally this was included in HOMEWORK_SYNONMS_LIST,
but removed and added here to avoid collision.

Example: http://stackoverflow.com/questions/1741820/assignment-operators-in-r-and
"""
# dict keys used; set as separate constants for easy access, and not having to rely on dict order
USER_MENU_OPTION_EXIT_KEY = "e"
USER_MENU_OPTION_HELP_KEY = "h"
USER_MENU_OPTION_ARG_KEY = "arg"
USER_MENU_OPTION_TYPE_KEY = "type"
USER_MENU_OPTION_NEW_PREDICTION = "p"
USER_MENU_OPTION_HELP_TEXT_KEY = "help"
USER_MENU_OPTION_ARGC_KEY = "arg_count"
USER_MENU_OPTION_LOAD_DEFAULT_KEY = "d"
USER_MENU_OPTION_METAVAR_KEY = "metavar"
USER_MENU_OPTION_LOAD_USER_MODEL_KEY = "l"
USER_MENU_OPTION_NEW_TRAINING_MODEL_KEY = "t"
USER_MENU_OPTION_CREATE_UNPROCESSED_DATASET = "u"
USER_MENU_OPTION_ARG_SELECTION_ARRAY = "arg_array"

USER_MENU_OPTIONS = {
    # create a new, unprocessed dataset
    USER_MENU_OPTION_CREATE_UNPROCESSED_DATASET: {
        USER_MENU_OPTION_HELP_TEXT_KEY: "Create an unprocessed data set based on database content "
                                        "(from database set in dbconfig.py). \n\tArguments: \n"
                                        "\tfilename: Filename for data set (model name will be the same as this) \n"
                                        "\tlimit: Limit for database row retrieval (integer) \n"
                                        "\tfeature_detectors: Create singular feature detectors based on data set? "
                                        "(Enter 0: No, 1: Yes) \n"
                                        "\tcreate_model: Create classifier model(s) based on data set? "
                                        "\n\t\t\t\t  0: No, 1: Unprocessed model, 2: Feature detector model(s) "
                                        "3: Both (1 and 2)",
        USER_MENU_OPTION_ARG_KEY: "--unprocessed",
        USER_MENU_OPTION_ARGC_KEY: 4,
        # required argument description
        USER_MENU_OPTION_METAVAR_KEY: ("filename", "limit", "feature_detectors", "create_model"),
        USER_MENU_OPTION_ARG_SELECTION_ARRAY: [0, 1, 2, 3]
    },
    # load a model that was created by the user
    USER_MENU_OPTION_LOAD_USER_MODEL_KEY: {
        # help menu displayed when using -h or --help
        USER_MENU_OPTION_HELP_TEXT_KEY: "Load user created model. Arguments: \n"
                                        "\tpath: Path to directory with model(s) (e.g. /home/user/my_models/) \n"
                                        "\tfilename: The models filename \n"
                                        "\tsuffix: File type - Optional (default: '.pkl')",
        # optional argument
        USER_MENU_OPTION_ARG_KEY: "--load-user-model",
        # amount of arguments required by this option
        USER_MENU_OPTION_ARGC_KEY: 2,
        # required argument description
        USER_MENU_OPTION_METAVAR_KEY: ("path",  "filename", "suffix")
    },
    # train a new model
    USER_MENU_OPTION_NEW_TRAINING_MODEL_KEY: {
        USER_MENU_OPTION_HELP_TEXT_KEY: "Train a new model based on an existing (or new) data set. Arguments: \n"
                                        "\tpath: Path to directory with training data (e.g. /home/user/my_data/) \n"
                                        "\tfilename: Filename for data set (model name will be the same as this) \n"
                                        "\tdb_load: Load from database (Enter 0: No, 1: Yes) \n"
                                        "\tlimit: Limit for database row retrieval (integer) - Optional unless "
                                        "'db_load' is '1'",
        USER_MENU_OPTION_ARG_KEY: "--train",
        USER_MENU_OPTION_ARGC_KEY: 3,
        # required argument description
        USER_MENU_OPTION_METAVAR_KEY: ("path",  "filename", "db_load", "limit")
    },
    # test out new prediction(s)
    USER_MENU_OPTION_NEW_PREDICTION: {
        USER_MENU_OPTION_HELP_TEXT_KEY: "Predict the quality of the entered question. Arguments: \n"
                                        "\tquestion: Question to predict quality of ",
        USER_MENU_OPTION_ARG_KEY: "--predict",
        USER_MENU_OPTION_ARGC_KEY: 1,
        # required argument description
        USER_MENU_OPTION_METAVAR_KEY: "predict_question"
    },
    # load default model (e.g. './training_data/training_data_10000.pkl)
    USER_MENU_OPTION_LOAD_DEFAULT_KEY: {
        USER_MENU_OPTION_HELP_TEXT_KEY: "Loads default model (if exists) from ./pickle_models",
        USER_MENU_OPTION_ARG_KEY: "--load-default-model",
        USER_MENU_OPTION_ARGC_KEY: 0,
        # required argument description
        USER_MENU_OPTION_METAVAR_KEY: "",
        USER_MENU_OPTION_TYPE_KEY: object,
    },
    # display the help
    USER_MENU_OPTION_HELP_KEY: {
        USER_MENU_OPTION_HELP_TEXT_KEY: "Displays this help menu",
        USER_MENU_OPTION_ARG_KEY: "--menu",
        USER_MENU_OPTION_ARGC_KEY: 0,
        # required argument description
        USER_MENU_OPTION_METAVAR_KEY: ""
    },
    # exit the program
    USER_MENU_OPTION_EXIT_KEY: {
        USER_MENU_OPTION_HELP_TEXT_KEY: "Exit the program",
        USER_MENU_OPTION_ARG_KEY: "--exit",
        USER_MENU_OPTION_ARGC_KEY: 0,
        # required argument description
        USER_MENU_OPTION_METAVAR_KEY: ""
    },
}
"""
Dictionary containing the options available upon program start
"""

CPU_COUNT = multiprocessing.cpu_count()
"""
Get the number of Logical CPUs for this computer
Used to adjust n_jobs for Windows systems
"""
