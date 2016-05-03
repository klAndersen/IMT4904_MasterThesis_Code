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


def create_new_training_model():
    global __so_dataframe
    global __classifier_model
    # TODO: Create a new model based on passed parameters
    # path, name, limit, classifier_data
    pass


def predict_question_quality(model, question):
    # TODO: Make a prediction based on loaded model and entered question
    # model, question (needs to be reconstructed to a string)
    # question would then need to be processed and then controlled against model
    pass


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
            question = args
            predict_question_quality(__classifier_model, question)
    elif command == USER_MENU_OPTION_LOAD_DEFAULT_KEY:
        limit = DATABASE_LIMIT.get('10000')
        model_name = "svm_detector_split_" + str(limit)
        dataset_file = FILEPATH_TRAINING_DATA + str(limit)
        __so_dataframe, __classifier_model = load_classifier_model_and_dataframe(model_name, dataset_file, limit)
        print_classifier_results(__classifier_model)
    elif command == USER_MENU_OPTION_LOAD_USER_MODEL_KEY:
        print(len(args))
    elif command == USER_MENU_OPTION_NEW_TRAINING_MODEL_KEY:
        create_new_training_model()
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
