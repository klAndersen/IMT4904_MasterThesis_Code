"""
Main entry file, this is the file to use when starting the program
"""

import constants as const
from time import time, ctime
from train_classifier import print_classifier_results, create_and_save_model
from file_processing import get_training_model, load_training_data, \
    create_and_save_feature_detectors, create_unprocessed_dataset_dump, create_singular_feature_detector_model
from text_processor import stem_training_data, process_question_for_prediction

__so_dataframe = None
__classifier_model = None


def create_unprocessed_dataset(args=list):
    """
    Creates a new classifier model based on the given filename

    Arguments:
        args (list): List containing the inputs used to create the new training model

    Returns:
         tuple(pandas.DataFrame, sklearn.model_selection._search.GridSearchCV):
         DataFrame containing the data set that was used, and the created model

    """
    # u tex_dump_ 10000 1 3
    # u training_data_ 10000 1 3
    model = None
    dataframe = None
    temp_dict = const.USER_MENU_OPTIONS.get(const.USER_MENU_OPTION_CREATE_UNPROCESSED_DATASET)
    argc = temp_dict.get(const.USER_MENU_OPTION_ARGC_KEY)
    arg_array = temp_dict.get(const.USER_MENU_OPTION_ARG_SELECTION_ARRAY)
    try:
        if (args is not None) and (len(args) == argc):
            filename = str(args[0])
            limit = int(args[1])
            create_feature_detectors = bool(int(args[2]))
            create_model = int(args[3])
            # if value is out of range, set it to min/max value
            if create_model > arg_array[3]:
                create_model = arg_array[3]
            elif create_model < arg_array[0]:
                create_model = arg_array[0]
            print("Loading data set...")
            # retrieve data from database, and create model if desired
            tags_filename = filename
            filename += str(limit) + "_unprocessed"
            dataframe = create_unprocessed_dataset_dump(filename, limit, tags_filename)
            # should a classifier model be created?
            if create_model == arg_array[3] or create_model == arg_array[1]:
                model = create_new_classifier_model(filename, dataframe, False)
            # should feature detectors be created?
            if create_feature_detectors:
                if create_model == arg_array[3] or create_model == arg_array[2]:
                    create_and_save_feature_detectors(filename, True, limit, tags_filename)
                else:
                    create_and_save_feature_detectors(filename, False, limit, tags_filename)
        else:
            missing_args = argc
            if args is not None:
                missing_args -= len(args)
            print("Missing " + str(missing_args) + " argument(s): \n",
                  temp_dict.get(const.USER_MENU_OPTION_HELP_TEXT_KEY))
    except ValueError as err:
        print(err)
    return dataframe, model


def create_new_classifier_model(filename, dataframe, stem_data=True, use_sgd_settings=False):
    """
    Creates a new classifier model based on the data in the passed ```dataframe```

    Arguments:
        filename (str): Filename for the model
        dataframe (pandas.DataFrame): DataFrame containing data to train classifier with
        stem_data (bool): Should the data set be stemmed before model is created?
        use_sgd_settings (bool): If True, run exhaustive SGD search. If False, use exhaustive SVC

    Returns:
         model (sklearn.model_selection._search.GridSearchCV): The created classifier model

    """
    print("Retrieving questions and classification labels...")
    training_data = dataframe[const.QUESTION_TEXT_KEY].copy()
    class_labels = dataframe[const.CLASS_LABEL_KEY].copy()
    if stem_data:
        print("Stemming questions...")
        index = 0
        for question in training_data:
            training_data[index] = stem_training_data(question)
            index += 1
    current_time("Starting training of model")
    if use_sgd_settings:
        filename += "_sgd"
    model_path = const.FILEPATH_MODELS + filename + ".pkl"
    model = create_and_save_model(training_data, class_labels, model_path, predict_proba=True,
                                  test_size=float(0.2), random_state=0, print_results=True,
                                  use_sgd_settings=use_sgd_settings)
    return model


def print_startup_menu():
    """
    Prints the startup menu displayed to the user on first run
    """
    menu = "Menu: \n"
    for key in sorted(const.USER_MENU_OPTIONS):
        temp_dict = const.USER_MENU_OPTIONS.get(key)
        menu += key + ": " + temp_dict.get(const.USER_MENU_OPTION_HELP_TEXT_KEY) + "\n"
    print(menu)


def end_program():
    print("Exiting program...")
    exit(0)


def check_path_ending(path):
    """
    Checks if path has a ending parameter

    To avoid issues with wrong paths, and since many functions rely on split
    paths and file names, this was added to account for missing ending for
    file paths. Example: '/home/lucas/my_folder' vs '/home/lucas/my_folder/'.
    This is achieved by using sys.platform.startswith().

    Arguments:
        path (str): Path to check

    Returns:
        str: path or modified path, where an '/' or '\\' has been added
    """
    sign = const.get_platform_separator()
    path_length = len(path)-1
    last_char = path[path_length]
    if last_char != sign:
        path += sign
    return path


def train_new_classifier_model(args=list):
    """
    Creates a new classifier model based on the input from the user

    Arguments:
        args (list): List containing the inputs used to create the new classifier model

    Returns:
         tuple(pandas.DataFrame, sklearn.model_selection._search.GridSearchCV):
         DataFrame containing the data set that was used, and the created model

    """
    limit = -1
    model = None
    dataframe = None
    create_additional_models = False
    temp_dict = const.USER_MENU_OPTIONS.get(const.USER_MENU_OPTION_NEW_TRAINING_MODEL_KEY)
    argc = temp_dict.get(const.USER_MENU_OPTION_ARGC_KEY)
    try:
        if (args is not None) and (len(args) >= argc):
            path = str(args[0])
            filename = str(args[1])
            db_load = bool(int(args[2]))
            path = check_path_ending(path)
            dataset_file = path + filename
            if db_load and len(args) > argc:
                limit = int(args[3])
                filename += str(limit)
                dataset_file += str(limit)
            elif db_load and len(args) == argc:
                raise(ValueError("Missing required parameter: Limit. When loading from database, "
                                 "amount of rows to retrieve is required."))
            elif not db_load and (len(args) > argc) and (args[3] == "eu"):
                # temporary easter egg
                create_additional_models = True
            print("Loading data set...")
            # create the training data set
            dataframe = load_training_data(dataset_file, db_load, limit, load_tags_from_database=False,
                                           exclude_site_tags=True, exclude_assignment=True)
            print("Data set loaded")
            if create_additional_models:
                # due to the time it takes to create new menu options, check them and control for
                # errors, this was added to as a temporary feature to be able to run proper feature comparison
                # depending on remaining time, this will most likely just be an easter egg
                print("Easter egg unlocked: Now creating two additional classifier models.")
                print("Creating model based on the best settings for the unprocessed data set...")
                file = filename + "_UP_settings"
                existing_model = filename + "_unprocessed"
                create_singular_feature_detector_model(file, existing_model, dataframe)
                print("Creating model without stemming, using exhaustive search...")
                file = filename + "_no_stem"
                model = create_new_classifier_model(file, dataframe, stem_data=False, use_sgd_settings=False)
            model = create_new_classifier_model(filename, dataframe, stem_data=True, use_sgd_settings=False)
        else:
            missing_args = argc
            if args is not None:
                missing_args -= len(args)
            print("Missing " + str(missing_args) + " argument(s): \n",
                  temp_dict.get(const.USER_MENU_OPTION_HELP_TEXT_KEY))
    except ValueError as err:
        print(err)
    return dataframe, model


def predict_question_quality(model, question):
    """
    Predicts the quality of the question based on the classifier model

    The question is processed by looking for potential features, and thereafter
    affixes is removed and stemmed. The model then predicts the accuracy of the
    question, which is printed on screen with the probability score and whether or
    not it was considered a good or bad question.

     Arguments:
        model (sklearn.model_selection._search.GridSearchCV): Classifier model
        question (str): Question to predict quality of

    """
    can_predict_probability = False
    processed_question = question.lower()
    prob_score = pred_prob_good = pred_prob_bad = -1
    processed_question = process_question_for_prediction(processed_question)
    # to be able to predict the probability for each class, it needs 'predict_proba' and 'probability=True'
    if hasattr(model, "predict_proba") and model.best_estimator_.get_params('probability'):
        pred_prob_bad = model.predict_proba([processed_question])[0][0]
        pred_prob_good = model.predict_proba([processed_question])[0][1]
        can_predict_probability = True
    # get the predicted class label and convert it to text
    predicted_class = model.predict([processed_question])[0]
    if predicted_class == -1:
        question_type = "bad"
        if can_predict_probability:
            prob_score = pred_prob_bad * 100
    else:
        question_type = "good"
        if can_predict_probability:
            prob_score = pred_prob_good * 100
    # print the results
    result_msg = "Your question is predicted to be a " + question_type + " question"
    if prob_score > -1:
        result_msg += ", with a probability of {0:.2f}%".format(prob_score)
    result_msg += "."
    print(result_msg)


def load_user_defined_model(args=list):
    """
    Loads a user defined model based on entered input.

    Arguments:
        args (list): User input arguments (expected: 3)
                     3 arguments; path, filename and file-suffix

    Returns:
        sklearn.model_selection._search.GridSearchCV: Trained classifier model at given location

    """
    model = None
    temp_dict = const.USER_MENU_OPTIONS.get(const.USER_MENU_OPTION_LOAD_USER_MODEL_KEY)
    argc = temp_dict.get(const.USER_MENU_OPTION_ARGC_KEY)
    if (args is not None) and (len(args) >= argc):
        path = str(args[0])
        path = check_path_ending(path)
        filename = str(args[1])
        # check if suffix was added
        if len(args) > argc:
            suffix = str(args[2])
            model = get_training_model(path, filename, suffix)
        else:
            model = get_training_model(path, filename)
    else:
        missing_args = argc
        if args is not None:
            missing_args -= len(args)
        print("Missing " + str(missing_args) + " argument(s): \n", temp_dict.get(const.USER_MENU_OPTION_HELP_TEXT_KEY))
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
        if u_input[0] == const.USER_MENU_OPTION_NEW_PREDICTION:
            command, args = u_input.split(" ", 1)
        else:
            command, *args = u_input.split()
    else:
        command = u_input
    if command == const.USER_MENU_OPTION_HELP_KEY:
        print_startup_menu()
    elif command == const.USER_MENU_OPTION_NEW_PREDICTION:
        if __classifier_model is None:
            print("No model is loaded. Load default model by entering 'd' or 'l path filename suffix'.")
        else:
            if args is not None:
                predict_question_quality(__classifier_model, args)
            else:
                print("No question entered. Please enter a question to predict.")
    elif command == const.USER_MENU_OPTION_LOAD_DEFAULT_KEY:
        limit = const.DATABASE_LIMIT.get('10000')
        default_name = "training_data_"
        model_name = default_name + str(limit) + "_unprocessed"
        dataset_file = const.FILEPATH_TRAINING_DATA + default_name + str(limit) + "_unprocessed"
        __so_dataframe = load_training_data(dataset_file, False, limit, tags_filename=default_name)
        __classifier_model = get_training_model(const.FILEPATH_MODELS, model_name)
        if __classifier_model is not None:
            print_classifier_results(__classifier_model)
    elif command == const.USER_MENU_OPTION_LOAD_USER_MODEL_KEY:
        if args is not None:
            __classifier_model = load_user_defined_model(args)
        else:
            temp_dict = const.USER_MENU_OPTIONS.get(const.USER_MENU_OPTION_LOAD_USER_MODEL_KEY)
            print("Missing argument(s): \n", temp_dict.get(const.USER_MENU_OPTION_HELP_TEXT_KEY))
        # was the model loaded?
        if __classifier_model is not None:
            print_classifier_results(__classifier_model)
    elif command == const.USER_MENU_OPTION_NEW_TRAINING_MODEL_KEY:
        if args is not None:
            __so_dataframe, __classifier_model = train_new_classifier_model(args)
        else:
            temp_dict = const.USER_MENU_OPTIONS.get(const.USER_MENU_OPTION_NEW_TRAINING_MODEL_KEY)
            print("Missing argument(s): \n", temp_dict.get(const.USER_MENU_OPTION_HELP_TEXT_KEY))
    elif command == const.USER_MENU_OPTION_CREATE_UNPROCESSED_DATASET:
        __so_dataframe, __classifier_model = create_unprocessed_dataset(args)
        if __classifier_model is not None:
            print_classifier_results(__classifier_model)
    elif command != const.USER_MENU_OPTION_EXIT_KEY:
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
    while user_input != const.USER_MENU_OPTION_EXIT_KEY:
        try:
            user_input = input("Enter option followed by arguments (if any). Enter h to show options: ")
            handle_user_input(user_input)
        except ValueError as err:
            print("ValueError: ", err)
        except OSError as err:
            print("OSError: ", err)
        except AttributeError as err:
            print("AttributeError: ", err)
        except TypeError as err:
            print("TypeError: ", err)
    if user_input == const.USER_MENU_OPTION_EXIT_KEY:
        end_program()


if __name__ == "__main__":
    main()
