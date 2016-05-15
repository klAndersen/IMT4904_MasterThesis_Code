"""
Handles all operations related to reading and writing to file (including database retrieval)
"""

import pickle
import constants
import text_processor
from os import listdir
from pandas import DataFrame
from os.path import isfile, join
from mysqldatabase import MySQLDatabase
from sklearn.externals.joblib import Memory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import dump_svmlight_file, load_svmlight_file


mem = Memory("./mem_cache")


def load_pickle_model(file_name=str):
    with open(file_name, 'rb') as file:
        model = pickle.load(file)
    return model


def dump_pickle_model(data, file_name=str):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


def get_training_model(path=str, model_name=str, suffix=".pkl"):
    """
    Returns the model with the given name at the given path if it exists

    Arguments:
        path (str): The directory path containing the training model(s)
        model_name (str): The filename of the model to retrieve
        suffix (str): The models file type (expects pickle type)

    Returns:
        model: The loaded pickle model || None if error occurred

    """
    try:
        # retrieve only the files with given suffix
        selected_files_only = [
            file for file in listdir(path)
            if (isfile(join(path, file)) and file.endswith(suffix))
            ]
        # was there any files in the given path, and does the model exists?
        if len(selected_files_only) > 0:
            file = model_name + suffix
            files_found = '\n'.join(map(lambda f: f, selected_files_only))
            if file in selected_files_only:
                file = path + file
                model = load_pickle_model(file)
                if model is not None:
                    feedback = "Model '" + model_name + suffix + "' loaded."
                    print(feedback)
                    return model
                else:
                    feedback = "Could not load the model '" + model_name + suffix + "'."
            else:
                feedback = "No classifier model named '" + model_name + suffix + "' found in '" + path + "."
            if feedback is not None:
                feedback += "\nThe following models were found: \n" + files_found
                print(feedback)
        else:
            feedback = "No classifier models found in '" + path + "."
            print(feedback)
    except Exception as ex:
        print("Failed at loading training model: ", ex)
    return None


@mem.cache
def load_training_data(file_location=str, load_from_database=False, limit=int(1000), return_svmlight=False,
                       create_feature_detectors=False, create_unprocessed=False, load_tags_from_database=False,
                       exclude_site_tags=False, exclude_assignment=False):
    """
    Loads training data either from database (if ```load_from_database``` is True) or from file

    This function loads training data from file (file must exist) or from database (requires database to exist).
    Depending on what sort of data you are after, there are three different calls you can make through this function.

    1.  Create an unprocessed training set, without any feature detectors or HTML cleaning (data as-is in database).
        To achieve this, set  ```create_unprocessed``` to True.
    2.  Create feature detectors. This creates separate files, where each file contains the named feature detector.
        To achieve this, set ```create_feature_detectors``` to True.
    3.  Create a training data set, where all HTML has been removed, and all current feature detectors have been added.
        To achieve this, set both ```create_feature_detectors``` and ```create_unprocessed``` to False.

    Arguments:
        file_location (str): Path + filename of files to save/load (e.g. 'path/training_data')
        load_from_database (bool): Should data be retrieved from database?
        limit (int): Amount of records to retrieve from database (default=1000)
        return_svmlight (bool): Should ```sklearn.datasets.load_svmlight_file``` be returned?
        create_feature_detectors (bool): Is this function being called to create feature detectors?
        create_unprocessed (bool): Is this function being called to create a clean, unprocessed dataset?
        load_tags_from_database (bool): Should site tags be loaded (only needed when loading dataset from database)?
        exclude_site_tags (bool): Should the site tags be excluded from feature detection?
        exclude_assignment (bool): Should 'assignment' words be excluded from feature detection?

    Returns:
         (pandas.DataFrame.from_csv, sklearn.datasets.load_svmlight_file):
         Tuple containing a pandas.DataFrame (all data retrieved from database) and
         tuple with training data (load_svmlight_file).
         If ```return_svmlight``` is set to False, only pandas.DataFrame.from_csv is returned

    See:
        | ```MySQLDatabase().retrieve_training_data```
        | ```pandas.DataFrame.to_csv```
        | ```pandas.DataFrame.from_csv```
        | ```sklearn.datasets.dump_svmlight_file```
        | ```sklearn.datasets.load_svmlight_file```
    """
    svm_file = file_location + ".dat"
    csv_file = file_location + ".csv"
    if load_from_database:
        if not exclude_site_tags:
            site_tags = load_tags(load_tags_from_database)
        else:
            site_tags = None
        comment = u"label: (-1: Bad question, +1: Good question); features: (term_id, frequency)"
        MySQLDatabase().set_vote_value_params()
        data = MySQLDatabase().retrieve_training_data(limit, create_feature_detectors, create_unprocessed,
                                                      site_tags, exclude_site_tags, exclude_assignment)
        # create a term-document matrix
        vectorizer = CountVectorizer(analyzer='word', min_df=0.01, stop_words="english")
        td_matrix = vectorizer.fit_transform(data.get(constants.QUESTION_TEXT_KEY))
        data.to_csv(csv_file)
        dump_svmlight_file(td_matrix, data[constants.CLASS_LABEL_KEY], f=svm_file, comment=comment)
    if not return_svmlight:
        return DataFrame.from_csv(csv_file)
    return DataFrame.from_csv(csv_file), load_svmlight_file(svm_file)


def load_tags(load_from_database=False):
    """
    Loads tags either from database (if ```load_from_database``` is True), else loads from file (presuming it exists)

    Arguments:
        load_from_database (bool): Should tags be loaded from the database?

    Returns:
         list: List containing the tags retrieved from the database
    """
    csv_file = constants.FILEPATH_TRAINING_DATA + "Tags.csv"
    if load_from_database:
        tag_data = MySQLDatabase().retrieve_all_tags()
        tag_data.to_csv(csv_file)
    else:
        tag_data = DataFrame.from_csv(csv_file)
    # convert DataFrame to list, and sort list based on tag length
    tag_list = tag_data[constants.TAG_NAME_COLUMN].tolist()
    tag_list.sort(key=len, reverse=True)
    return tag_list


@mem.cache
def __create_and_save_feature_detectors(limit=int(1000)):
    """
    Creates singular feature files where the amount of rows retrieved is equal to limit.
    The files are stored in ./feature_detectors

    The following feature detectors are created:
    - has_codeblock: Replaces code blocks with this text
    - has_link: Replaces links with this text
    - has_homework & has_assignment: Replaces "homework words" with this text
    - has_numeric: Replaces numerical values with this text
    - has_hexadecimal: Replaces hexadecimal values with this text

    Arguments:
        limit (int): Amount of rows to retrieve from database

    """
    csv_file = constants.FILEPATH_TRAINING_DATA + str(limit) + "_unprocessed.csv"
    file_location = constants.FILEPATH_FEATURE_DETECTOR + str(limit) + "_"
    try:
        data_copy = DataFrame.from_csv(csv_file)
    except OSError:
        feedback_msg = "Could not find unprocessed data set. File:" + csv_file \
                       + ". \nAttempting to retrieve from Database:"
        print(feedback_msg)
        MySQLDatabase().set_vote_value_params()
        __create_unprocessed_dataset_dump(limit)
        feedback_msg = "Data loaded successfully!"
        print(feedback_msg)
        data_copy = DataFrame.from_csv(csv_file)
    # create feature detector for code blocks
    feedback_msg = "Creating singular feature detector: "
    print(feedback_msg, "Code blocks")
    filename = constants.QUESTION_HAS_CODEBLOCK_KEY
    __create_and_save_feature_detector_html(text_processor.__set_has_codeblock, file_location, filename, data_copy)
    # create feature detector for links
    print(feedback_msg, "Links")
    data_copy = DataFrame.from_csv(csv_file)
    filename = constants.QUESTION_HAS_LINKS_KEY
    __create_and_save_feature_detector_html(text_processor.__set_has_link, file_location, filename, data_copy)
    # create feature detector for has_homework & has_assignment
    print(feedback_msg, "Homework")
    data_copy = get_processed_dataset(csv_file)
    filename = constants.QUESTION_HAS_HOMEWORK_KEY
    __create_and_save_feature_detector_homework(file_location, filename, data_copy)
    # create feature detector for has_numeric
    print(feedback_msg, "Numerical")
    data_copy = get_processed_dataset(csv_file)
    filename = constants.QUESTION_HAS_NUMERIC_KEY
    __create_and_save_feature_detector(text_processor.__set_has_numeric, file_location, filename, data_copy)
    # create feature detector for has_hexadecimal
    print(feedback_msg, "Hexadecimal")
    data_copy = get_processed_dataset(csv_file)
    filename = constants.QUESTION_HAS_HEXADECIMAL_KEY
    __create_and_save_feature_detector(text_processor.__set_has_hexadecimal, file_location, filename, data_copy)
    # create feature detector for tags
    filename = "has_tags"
    print(feedback_msg, "Tags")
    data_copy = get_processed_dataset(csv_file)
    __create_and_save_feature_detectors_tags(file_location, filename, data_copy)


def get_processed_dataset(csv_filename):
    index = 0
    dataset = DataFrame.from_csv(csv_filename)
    # convert all the HTML to normal text
    for question in dataset[constants.QUESTION_TEXT_KEY]:
        question = text_processor.remove_html_tags_from_text(question, False)
        question = question.lower()
        dataset.loc[index, constants.QUESTION_TEXT_KEY] = question
        index += 1
    return dataset


def __create_and_save_feature_detector(exec_function, file_location=str, filename=str, training_data=DataFrame):
    """
    Creates feature detectors from text where only one parameter (question text) is needed.
    Requires processed question text, without HTML. Saves to file afterwards.

    Arguments:
        exec_function: Function to execute to create features
        file_location (str): The path to the file
        filename (str): Name of the file (e.g. "feature_detector")
        training_data (pandas.DataFrame):DataFrame containing all related data, where Questions doesn't contain HTML

    """
    index = 0
    # loop through the questions and extract related features
    for question in training_data[constants.QUESTION_TEXT_KEY]:
        training_data.loc[index, constants.QUESTION_TEXT_KEY] = exec_function(question)
        index += 1
    # save to file
    filename = file_location + filename + ".csv"
    training_data.to_csv(filename)


def __create_and_save_feature_detector_homework(file_location=str, filename=str, training_data=DataFrame):
    """
    Since homework and assignment are separated (but still considered as homework), these are handled here
    Requires processed, non-HTML text. Saves to file afterwards.

    Arguments:
        file_location (str): The path to the file
        filename (str): Name of the file (e.g. "feature_detector")
        training_data (pandas.DataFrame):DataFrame containing all related data, where Questions doesn't contain HTML

    """
    index = 0
    assignment_list = constants.ASSIGNMENT_LIST
    homework_list = constants.HOMEWORK_SYNONMS_LIST
    homework_list.sort(key=len, reverse=True)
    has_homework = constants.QUESTION_HAS_HOMEWORK_KEY
    has_assignment = constants.QUESTION_HAS_ASSIGNMENT_KEY
    # loop through questions to find homework and its synonyms
    for question in training_data[constants.QUESTION_TEXT_KEY]:
        question = text_processor.__set_has_homework_or_assignment(question, has_homework, homework_list)
        updated_question = text_processor.__set_has_homework_or_assignment(question, has_assignment, assignment_list)
        training_data.loc[index, constants.QUESTION_TEXT_KEY] = updated_question
        index += 1
    # save to file
    filename = file_location + filename + ".csv"
    training_data.to_csv(filename)


def __create_and_save_feature_detector_html(exec_function, file_location=str, filename=str, training_data=DataFrame):
    """
    Creates feature detector for those that require HTML tags to be properly extracted,
    afterwards it converted to normal text and saved to file

    Arguments:
        exec_function: Function to execute to create features
        file_location (str): The path to the file
        filename (str): Name of the file (e.g. "feature_detector")
        training_data (pandas.DataFrame): DataFrame containing all related data, where the Questions are wrapped in HTML

    """
    index = 0
    # loop through the questions and extract related features
    for question in training_data[constants.QUESTION_TEXT_KEY]:
        question = exec_function(question)
        training_data.loc[index, constants.QUESTION_TEXT_KEY] = question
        index += 1

    index = 0
    # loop through the questions and convert HTML to normal text
    for question in training_data[constants.QUESTION_TEXT_KEY]:
        question = text_processor.remove_html_tags_from_text(question, False)
        training_data.loc[index, constants.QUESTION_TEXT_KEY] = question
        index += 1
    # save to file
    filename = file_location + filename + ".csv"
    training_data.to_csv(filename)


def __create_and_save_feature_detectors_tags(file_location=str, filename=str, training_data=DataFrame):
    """
    Retrieves tags from the question and the database, and uses this to extract and replace tags in the question.
    Requires Questions without HTML. Saves data to file.

    Arguments:
        file_location (str): The path to the file
        filename (str): Name of the file (e.g. "feature_detector")
        training_data (pandas.DataFrame): DataFrame containing all related data, where Questions doesn't contain HTML

    """
    index = 0
    tag_data = None
    try:
        suffix = ".csv"
        path = "./training_data"
        # retrieve only the files with given suffix
        selected_files_only = [
            file for file in listdir(path)
            if (isfile(join(path, file)) and file.endswith(suffix))
            ]
        # was there any files in the given path, and does the file exists?
        if len(selected_files_only) > 0:
            file = "Tags" + suffix
            if file in selected_files_only:
                file = constants.FILEPATH_TRAINING_DATA + file
                tag_data = DataFrame.from_csv(file)
    except Exception as ex:
        print("Failed at loading Tags file: ", ex)
    # was the Tags data successfully loaded from file?
    if tag_data is None:
        tag_data = MySQLDatabase().retrieve_all_tags()
    text_tags = training_data["Tags"].tolist()
    text_tags.sort(key=len, reverse=True)
    text_tags = text_processor.process_tags(text_tags)
    site_tags = tag_data[constants.TAG_NAME_COLUMN].tolist()
    site_tags.sort(key=len, reverse=True)
    for question in training_data[constants.QUESTION_TEXT_KEY]:
        question = text_processor.__set_has_tag(question, text_tags[index], site_tags)
        training_data.loc[index, constants.QUESTION_TEXT_KEY] = question
        index += 1
    # save to file
    filename = file_location + filename + ".csv"
    training_data.to_csv(filename)


def __create_unprocessed_dataset_dump(limit=int(1000)):
    """
    Creates a clean dataset without any form of processing.

    What this means is that the content is -exactly- as it is in the database,
    without HTML removal, feature detector replacement or labels. However,
    it does not contain all the columns, only those columns which are used in
    the processed training sets.

    Arguments:
        limit (int): Amount of rows to retrieve from database

    """
    file_location = constants.FILEPATH_TRAINING_DATA + str(limit) + "_unprocessed"
    load_training_data(file_location, True, limit, create_unprocessed=True)

# __create_and_save_feature_detectors(10000)
