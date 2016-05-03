"""
File containing constants used in more than one place
"""

from re import compile, IGNORECASE, VERBOSE

DATABASE_LIMIT = {
    '10': 10,
    '100': 100,
    '1000': 1000,
    '10000': 10000
}

# dict keys used; set as separate constants for easy access, and not having to rely on dict order
USER_MENU_OPTION_EXIT_KEY = "-e"
USER_MENU_OPTION_HELP_KEY = "-m"
USER_MENU_OPTION_ARG_KEY = "arg"
USER_MENU_OPTION_TYPE_KEY = "type"
USER_MENU_OPTION_NEW_PREDICTION = "-p"
USER_MENU_OPTION_HELP_TEXT_KEY = "help"
USER_MENU_OPTION_ARGC_KEY = "arg_count"
USER_MENU_OPTION_LOAD_DEFAULT_KEY = "-d"
USER_MENU_OPTION_METAVAR_KEY = "metavar"
USER_MENU_OPTION_LOAD_USER_MODEL_KEY = "-l"
USER_MENU_OPTION_NEW_TRAINING_MODEL_KEY = "-t"

USER_MENU_OPTIONS = {
    # load a model that was created by the user
    USER_MENU_OPTION_LOAD_USER_MODEL_KEY: {
        # help menu displayed when using -h or --help
        USER_MENU_OPTION_HELP_TEXT_KEY: "Load user created model \n"
                                        "\tpath: Path to directory with model(s) \n"
                                        "\tfilename: The models filename \n"
                                        "\tsuffix: File type (e.g. '.pkl')",
        # optional argument
        USER_MENU_OPTION_ARG_KEY: "--load-user-model",
        # amount of arguments required by this option
        USER_MENU_OPTION_ARGC_KEY: 3,
        # required argument description
        USER_MENU_OPTION_METAVAR_KEY: ("path",  "filename", "suffix")
    },
    # train a new model
    USER_MENU_OPTION_NEW_TRAINING_MODEL_KEY: {
        USER_MENU_OPTION_HELP_TEXT_KEY: "Train a new model \n"
                                        "\tpath: Path to directory with model(s) \n"
                                        "\tfilename: The models filename \n"
                                        "\tdb_load: Load from database (y/n) \n"
                                        "\tlimit: Database limit (integer)",
        USER_MENU_OPTION_ARG_KEY: "--train",
        USER_MENU_OPTION_ARGC_KEY: 4,
        # required argument description
        USER_MENU_OPTION_METAVAR_KEY: ("path",  "filename", "db_load", "limit")
    },
    # test out new prediction(s)
    USER_MENU_OPTION_NEW_PREDICTION: {
        USER_MENU_OPTION_HELP_TEXT_KEY: "Predict the quality of the entered question",
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

FILEPATH_TRAINING_DATA = "./training_data/training_data_"
"""
The path to where the training data can be found
"""

FILEPATH_MODELS = "./pickle_models/"
"""
The path to were the produced models can be found
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
