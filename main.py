"""
Main entry file, this is the file to use when starting the program
"""

from constants import *
from time import time, ctime
from file_processing import *
from train_classifier import print_classifier_results

__so_dataframe = None
__classifier_model = None


def print_startup_menu():
    """
    Prints the startup menu displayed to the user on first run
    """
    menu = "Menu: \n"
    for key in sorted(USER_MENU_OPTIONS):
        temp_dict = USER_MENU_OPTIONS.get(key)
        menu += key + ": " + temp_dict.get(USER_MENU_OPTION_HELP_TEXT_KEY) + "\n"
    print(menu)


def end_program():
    print("Exiting program...")
    exit(0)


def create_new_training_model(args=list):
    """
    Creates a new classifier model based on the given filename

    Arguments:
        args (list): List containing the inputs used to create the new training model

    Returns:
         tuple(pandas.DataFrame, model): DataFrame containing the data set that was used, and the created model

    """
    model = None
    dataframe = None

    temp_dict = USER_MENU_OPTIONS.get(USER_MENU_OPTION_NEW_TRAINING_MODEL_KEY)
    argc = temp_dict.get(USER_MENU_OPTION_ARGC_KEY)
    if (args is not None) and (len(args) == argc):
        path = str(args[0])
        filename = str(args[1])
        db_load = bool(args[2])
        limit = int(args[3])
        dataset_file = path + filename + str(limit)
        # create the training data set
        dataframe = load_training_data(dataset_file, db_load, limit, load_tags_from_database=False,
                                       exclude_site_tags=True, exclude_assignment=True)
        # TODO: add function for training and creating new model
        '''
        1. Get questions and labels from dataframe
        2. Lemmatize and stem questions
        3. Set location for where to save model
        4. Train model with set parameters (default)
        5. Print results
        '''
    else:
        missing_args = argc
        if args is not None:
            missing_args -= len(args)
        print("Missing " + str(missing_args) + " argument(s): \n", temp_dict.get(USER_MENU_OPTION_HELP_TEXT_KEY))
    return dataframe, model


def predict_question_quality(model, question):
    """

     Arguments:
        model:
        question (str): Question to predict quality of

    """
    # TODO: Make a prediction based on loaded model and entered question
    '''
    0. Add a note that code samples are not tested
    1. Load model (if not loaded; give error msg)
    2. Check if question has been entered
    3. Convert input to string (bcz args=array)
    4. Convert question to lower
    5. Run feature detection
    6. Lemmatize and stem question
    7. Pass question to model for prediction
    8. Print results
    '''
    pass


def load_user_defined_model(args=list):
    """
    Loads a user defined model based on entered input.

    Arguments:
        args (list): User input arguments (expected: 3)
                     3 arguments; path, filename and file-suffix

    Returns:
        model: Trained classifier model at given location

    """
    model = None
    temp_dict = USER_MENU_OPTIONS.get(USER_MENU_OPTION_LOAD_USER_MODEL_KEY)
    argc = temp_dict.get(USER_MENU_OPTION_ARGC_KEY)
    if (args is not None) and (len(args) == argc):
        path = str(args[0])
        filename = str(args[1])
        suffix = str(args[2])
        model = get_training_model(path, filename, suffix)
    else:
        missing_args = argc
        if args is not None:
            missing_args -= len(args)
        print("Missing " + str(missing_args) + " argument(s): \n", temp_dict.get(USER_MENU_OPTION_HELP_TEXT_KEY))
    return model


def handle_user_input(u_input=str):
    """
    Takes the user input and checks what operation to execute

    Arguments:
        u_input (str): The users input

    """
    global __so_dataframe
    global __classifier_model
    args = None
    if len(u_input) > 1:
        command, *args = u_input.split()
    else:
        command = u_input
    if command == USER_MENU_OPTION_HELP_KEY:
        print_startup_menu()
    elif command == USER_MENU_OPTION_NEW_PREDICTION:
        if __classifier_model is None:
            print("No model is loaded. Load default model by entering 'd' or 'l path filename suffix'.")
        else:
            if args is not None:
                question = args
                predict_question_quality(__classifier_model, question)
    elif command == USER_MENU_OPTION_LOAD_DEFAULT_KEY:
        limit = DATABASE_LIMIT.get('10000')
        model_name = "svm_detector_split_" + str(limit)
        dataset_file = FILEPATH_TRAINING_DATA + str(limit)
        __so_dataframe = load_training_data(dataset_file, False, limit)
        __classifier_model = get_training_model(constants.FILEPATH_MODELS, model_name)
        if __classifier_model is not None:
            print_classifier_results(__classifier_model)
    elif command == USER_MENU_OPTION_LOAD_USER_MODEL_KEY:
        if args is not None:
            __classifier_model = load_user_defined_model(args)
        else:
            temp_dict = USER_MENU_OPTIONS.get(USER_MENU_OPTION_LOAD_USER_MODEL_KEY)
            print("Missing argument(s): \n", temp_dict.get(USER_MENU_OPTION_HELP_TEXT_KEY))
        # was the model loaded?
        if __classifier_model is not None:
            print_classifier_results(__classifier_model)
    elif command == USER_MENU_OPTION_NEW_TRAINING_MODEL_KEY:
        if args is not None:
            create_new_training_model(args)
        else:
            temp_dict = USER_MENU_OPTIONS.get(USER_MENU_OPTION_NEW_TRAINING_MODEL_KEY)
            print("Missing argument(s): \n", temp_dict.get(USER_MENU_OPTION_HELP_TEXT_KEY))
    elif command != USER_MENU_OPTION_EXIT_KEY:
        print("Invalid command: ", command)


def current_time(info=str):
    """
    Prints the current time (Date HH:mm:ss)

    Arguments:
        info (str): Optional string containing info about this timestamp
    """
    time_now = ctime(time())
    if info is None:
        print(time_now)
    else:
        print(info, time_now)
    print('\n')


def main():
    user_input = ""
    current_time("Program started")
    print_startup_menu()
    while user_input != USER_MENU_OPTION_EXIT_KEY:
        try:
            user_input = input("Enter option followed by arguments (if any) Enter h to show options: ")
            handle_user_input(user_input)
        except ValueError as ex:
            print("ValueError: ", ex)
    if user_input == USER_MENU_OPTION_EXIT_KEY:
        end_program()


if __name__ == "__main__":
    main()
