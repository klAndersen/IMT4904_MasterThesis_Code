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
            feedback = "No classifier models found in '" + path + "'."
            print(feedback)
    except Exception as ex:
        print("Failed at loading training model: ", ex)
    return None


def create_unprocessed_dataset_dump(filename, limit=int(1000), tags_filename=""):
    """
    Creates a clean data set without any form of processing.

    What this means is that the content is -exactly- as it is in the database,
    without HTML removal, feature detector replacement or labels. However,
    it does not contain all the columns, only those columns which are used in
    the processed training sets.

    Arguments:
        filename (str): The name of the file
        limit (int): Amount of rows to retrieve from database
        tags_filename (str): Optional filename; e.g. if using multiple StackExchange sites this should be used

    Returns:
        pandas.DataFrame: DataFrame containing the unprocessed data loaded from database

    """
    file_location = constants.FILEPATH_TRAINING_DATA + filename
    return load_training_data(file_location, True, limit, create_unprocessed=True, tags_filename=tags_filename,
                              load_tags_from_database=True)


@mem.cache
def load_training_data(file_location=str, load_from_database=False, limit=int(1000),
                       create_feature_detectors=False, create_unprocessed=False, load_tags_from_database=False,
                       tags_filename="", exclude_site_tags=False, exclude_assignment=False):
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
        create_feature_detectors (bool): Is this function being called to create feature detectors?
        create_unprocessed (bool): Is this function being called to create a clean, unprocessed data set?
        load_tags_from_database (bool): Should site tags be loaded (only needed when loading data set from database)?
        tags_filename (str): Optional filename; e.g. if using multiple StackExchange sites this should be used
        exclude_site_tags (bool): Should the site tags be excluded from feature detection?
        exclude_assignment (bool): Should 'assignment' words be excluded from feature detection?

    Returns:
         pandas.DataFrame.from_csv: DataFrame containing the data set that was either loaded from database or file

    See:
        | ```MySQLDatabase().retrieve_training_data```
        | ```pandas.DataFrame.to_csv```
        | ```pandas.DataFrame.from_csv```
    """
    csv_file = file_location + ".csv"
    if not exclude_site_tags:
        site_tags = load_tags(tags_filename, load_tags_from_database)
    else:
        site_tags = None
    if load_from_database:
        mysqldb = MySQLDatabase()
        # mysqldb.set_vote_value_params()
        data = mysqldb.retrieve_training_data(limit, create_feature_detectors, create_unprocessed,
                                              site_tags, exclude_site_tags, exclude_assignment)
        data.to_csv(csv_file, encoding='utf-8')
        return data
    return DataFrame.from_csv(csv_file, encoding='utf-8')


def load_tags(tags_filename="", load_from_database=False):
    """
    Loads tags either from database (if ```load_from_database``` is True), else loads from file (presuming it exists)

    Arguments:
        tags_filename (str): Optional filename; e.g. if using multiple StackExchange sites this should be used
        load_from_database (bool): Should tags be loaded from the database?

    Returns:
         list: List containing the tags retrieved from the database
    """
    csv_file = constants.FILEPATH_TRAINING_DATA + tags_filename + "Tags.csv"
    if load_from_database:
        tag_data = MySQLDatabase().retrieve_all_tags()
        tag_data.to_csv(csv_file, encoding='utf-8')
    else:
        tag_data = DataFrame.from_csv(csv_file, encoding='utf-8')
    # convert DataFrame to list, and sort list based on tag length
    tag_list = tag_data[constants.TAG_NAME_COLUMN].tolist()
    tag_list.sort(key=len, reverse=True)
    return tag_list


@mem.cache
def create_and_save_feature_detectors(filename, create_models=False, limit=int(1000), site_tags_filename=str):
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
        filename (str): The name of the data set that contains the unprocessed data
        create_models (bool): Should models be created for these singular feature detector(s)?
        limit (int): Amount of rows to retrieve from database
        site_tags_filename (str): Name of file containing all the tags found at the given site

    Returns:
        pandas.DataFrame: DataFrame containing the loaded, unprocessed data set

    """
    file_location = constants.FILEPATH_FEATURE_DETECTOR + filename + "_"
    csv_file = constants.FILEPATH_TRAINING_DATA + filename + ".csv"
    try:
        data_copy = DataFrame.from_csv(csv_file, encoding='utf-8')
    except OSError:
        feedback_msg = "Could not find unprocessed data set. File:" + csv_file \
                       + ". \nAttempting to retrieve from Database..."
        print(feedback_msg)
        create_unprocessed_dataset_dump(filename, limit)
        feedback_msg = "Data loaded successfully!"
        print(feedback_msg)
        data_copy = DataFrame.from_csv(csv_file, encoding='utf-8')
    # create feature detector for code blocks
    feedback_msg = "Creating singular feature detector: "
    print(feedback_msg, "Code blocks")
    new_filename = constants.QUESTION_HAS_CODEBLOCK_KEY.strip()
    __create_and_save_feature_detector_html(text_processor.__set_has_codeblock, file_location,
                                            new_filename, data_copy, create_models, filename)
    # create feature detector for links
    print(feedback_msg, "Links")
    data_copy = DataFrame.from_csv(csv_file, encoding='utf-8')
    new_filename = constants.QUESTION_HAS_LINKS_KEY.strip()
    __create_and_save_feature_detector_html(text_processor.__set_has_link, file_location, new_filename,
                                            data_copy, create_models, filename)
    # create feature detector for has_homework & has_assignment
    print(feedback_msg, "Homework")
    data_copy = get_processed_dataset(csv_file)
    new_filename = constants.QUESTION_HAS_HOMEWORK_KEY.strip()
    __create_and_save_feature_detector_homework(file_location, new_filename, data_copy,
                                                create_models, filename)
    # create feature detector for has_numeric
    print(feedback_msg, "Numerical")
    data_copy = get_processed_dataset(csv_file)
    new_filename = constants.QUESTION_HAS_NUMERIC_KEY.strip()
    __create_and_save_feature_detector(text_processor.__set_has_numeric, file_location, new_filename,
                                       data_copy, create_models, filename)
    # create feature detector for has_hexadecimal
    print(feedback_msg, "Hexadecimal")
    data_copy = get_processed_dataset(csv_file)
    new_filename = constants.QUESTION_HAS_HEXADECIMAL_KEY.strip()
    __create_and_save_feature_detector(text_processor.__set_has_hexadecimal, file_location, new_filename,
                                       data_copy, create_models, filename)
    # create feature detector for tags
    new_filename = "has_tags"
    print(feedback_msg, "Tags")
    data_copy = get_processed_dataset(csv_file)
    __create_and_save_feature_detectors_tags(file_location, new_filename, data_copy, create_models,
                                             filename, site_tags_filename)


def get_processed_dataset(csv_filename):
    """

    Arguments:
        csv_filename (str): Name of CSV file containing data set to load

    Returns:
        pandas.DataFrame: DataFrame containing the data set loaded from file
    """
    index = 0
    dataset = DataFrame.from_csv(csv_filename, encoding='utf-8')
    # convert all the HTML to normal text
    for question in dataset[constants.QUESTION_TEXT_KEY]:
        question = text_processor.remove_html_tags_from_text(question, False)
        question = question.lower()
        dataset.loc[index, constants.QUESTION_TEXT_KEY] = question
        index += 1
    return dataset


def __create_and_save_feature_detector(exec_function, file_location=str, filename=str, training_data=DataFrame,
                                       create_model=False, unprocessed_model_name=str):
    """
    Creates feature detectors from text where only one parameter (question text) is needed.
    Requires processed question text, without HTML. Saves to file afterwards.

    Arguments:
        exec_function: Function to execute to create features
        file_location (str): The path to the file
        filename (str): Name of the file (e.g. "feature_detector")
        training_data (pandas.DataFrame):DataFrame containing all related data, where Questions doesn't contain HTML
        create_model (bool): Should a classifier model be created for this feature detector? Default: False
        unprocessed_model_name (str): Name of the data set which this feature detector is based for (```create_model```)

    """
    index = 0
    # loop through the questions and extract related features
    for question in training_data[constants.QUESTION_TEXT_KEY]:
        training_data.loc[index, constants.QUESTION_TEXT_KEY] = exec_function(question)
        index += 1
    # save to file
    file = file_location + filename + ".csv"
    training_data.to_csv(file, encoding='utf-8')
    if create_model:
        feature_model_name = unprocessed_model_name + "_" + filename
        create_singular_feature_detector_model(feature_model_name, unprocessed_model_name, training_data)


def __create_and_save_feature_detector_homework(file_location=str, filename=str, training_data=DataFrame,
                                                create_model=False, unprocessed_model_name=str):
    """
    Since homework and assignment are separated (but still considered as homework), these are handled here
    Requires processed, non-HTML text. Saves to file afterwards.

    Arguments:
        file_location (str): The path to the file
        filename (str): Name of the file (e.g. "feature_detector")
        training_data (pandas.DataFrame):DataFrame containing all related data, where Questions doesn't contain HTML
        create_model (bool): Should a classifier model be created for this feature detector? Default: False
        unprocessed_model_name (str): Name of the data set which this feature detector is based for (```create_model```)

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
    file = file_location + filename + ".csv"
    training_data.to_csv(file, encoding='utf-8')
    if create_model:
        feature_model_name = unprocessed_model_name + "_" + filename
        create_singular_feature_detector_model(feature_model_name, unprocessed_model_name, training_data)


def __create_and_save_feature_detector_html(exec_function, file_location=str, filename=str,
                                            training_data=DataFrame, create_model=False,
                                            unprocessed_model_name=str):
    """
    Creates feature detector for those that require HTML tags to be properly extracted,
    afterwards it converted to normal text and saved to file

    Arguments:
        exec_function: Function to execute to create features
        file_location (str): The path to the file
        filename (str): Name of the file (e.g. "feature_detector")
        training_data (pandas.DataFrame): DataFrame containing all related data, where the Questions are wrapped in HTML
        create_model (bool): Should a classifier model be created for this feature detector? Default: False
        unprocessed_model_name (str): Name of the data set which this feature detector is based for (```create_model```)

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
    file = file_location + filename + ".csv"
    training_data.to_csv(file, encoding='utf-8')
    if create_model:
        feature_model_name = unprocessed_model_name + "_" + filename
        create_singular_feature_detector_model(feature_model_name, unprocessed_model_name, training_data)


def __create_and_save_feature_detectors_tags(file_location=str, filename=str, training_data=DataFrame,
                                             create_model=False, unprocessed_model_name=str, site_tags_filename=str):
    """
    Retrieves tags from the question and the database, and uses this to extract and replace tags in the question.
    Requires Questions without HTML. Saves data to file.

    Arguments:
        file_location (str): The path to the file
        filename (str): Name of the file (e.g. "feature_detector")
        training_data (pandas.DataFrame): DataFrame containing all related data, where Questions doesn't contain HTML
        create_model (bool): Should a classifier model be created for this feature detector? Default: False
        unprocessed_model_name (str): Name of the data set which this feature detector is based for (```create_model```)
        site_tags_filename (str): Name of file containing all the tags found at the given site

    """
    index = 0
    tag_data = None
    try:
        suffix = ".csv"
        path = constants.FILEPATH_TRAINING_DATA
        # retrieve only the files with given suffix
        selected_files_only = [
            file for file in listdir(path)
            if (isfile(join(path, file)) and file.endswith(suffix))
            ]
        # was there any files in the given path, and does the file exists?
        if len(selected_files_only) > 0:
            file = site_tags_filename + "Tags" + suffix
            if file in selected_files_only:
                file = constants.FILEPATH_TRAINING_DATA + file
                tag_data = DataFrame.from_csv(file, encoding='utf-8')
    except Exception as ex:
        print("Failed at loading Tags file: ", ex)
    # was the Tags data successfully loaded from file?
    if tag_data is None:
        tag_data = MySQLDatabase().retrieve_all_tags()
    text_tags = training_data["Tags"].tolist()
    text_tags.sort(key=len, reverse=True)
    text_tags = text_processor.process_tags(text_tags)
    site_tags = tag_data[constants.TAG_NAME_COLUMN].tolist()
    # for some reason, some values are interpreted as float (e.g. 'nan'; index: 4413)
    # therefore, list comprehension is used to use a forced transformation to ensure all values are strings
    site_tags = sorted([str(tag) for tag in site_tags])
    # site_tags.sort(key=len(), reverse=True)
    for question in training_data[constants.QUESTION_TEXT_KEY]:
        question = text_processor.__set_has_tag(question, text_tags[index], site_tags)
        training_data.loc[index, constants.QUESTION_TEXT_KEY] = question
        index += 1
    # save to file
    file = file_location + filename + ".csv"
    training_data.to_csv(file, encoding='utf-8')
    if create_model:
        feature_model_name = unprocessed_model_name + "_" + filename
        create_singular_feature_detector_model(feature_model_name, unprocessed_model_name, training_data)


def create_singular_feature_detector_model(new_model_name, existing_model_name, dataframe):
    """
    Creates a new classifier model for a single feature based on values from existing model

    To be able to see if a new feature has any impact on accuracy and prediction, it is
    necessary to compare the accuracy before and after the feature was added. This is achieved
    by re-using the best estimation - and parameter values from the classifier model that
    was trained on the unprocessed data set. By doing this, one avoids that new optimal parameters
    are created, which could give a false positive on the score (since score could be improved
    due to the new parameter setting, and not the actual feature that was added).

    Arguments:
        new_model_name (str): The name of the classifier model to create
        existing_model_name (str): The name of the existing model to base the new model on
        dataframe (pandas.DataFrame): DataFrame containing the data set for training the new model

    """
    path = constants.FILEPATH_MODELS
    model = get_training_model(path, existing_model_name)
    if model is not None:
        pipeline_svm = model.best_estimator_
        # set up the parameter values
        param_svm = [
            {
                'clf__C': [model.best_params_['clf__C']],
                'clf__kernel': [model.best_params_['clf__kernel']],
             },
            ]
        # check if gamma is a part of the parameters
        if model.best_params_.get('clf__gamma') is not None:
            param_svm[0]['clf__gamma'] = [model.best_params_.get('clf__gamma')]
        print("Retrieving questions and classification labels...")
        training_data = dataframe[constants.QUESTION_TEXT_KEY].copy()
        class_labels = dataframe[constants.CLASS_LABEL_KEY].copy()
        print("Starting training of model")
        model_path = constants.FILEPATH_SINGULAR_FEATURE_MODELS + new_model_name + ".pkl"
        # to avoid circular import issue
        from train_classifier import create_singular_feature_detector_model
        create_singular_feature_detector_model(pipeline_svm, param_svm, model_path, training_data, class_labels,
                                               test_size=float(0.2), random_state=0)


def create_feature_detector_model(filename=str, dataframe=DataFrame, limit=int):
    """
    Creates a new classifier model for a singular feature detector

    (not used in this thesis, because it may have other estimator values which
    makes it not comparable to the unprocessed data set)

    Arguments:
        filename (str): Filename for the model
        dataframe (pandas.DataFrame): The dataframe containing the data for creating the model
        limit (int): The amount of rows

    """
    # to avoid circular import issue
    from train_classifier import create_and_save_model
    print("Retrieving questions and classification labels...")
    training_data = dataframe[constants.QUESTION_TEXT_KEY].copy()
    class_labels = dataframe[constants.CLASS_LABEL_KEY].copy()
    print("Starting training of model")
    model_path = constants.FILEPATH_SINGULAR_FEATURE_MODELS + filename + str(limit) + ".pkl"
    create_and_save_model(training_data, class_labels, model_path, predict_proba=True,
                          test_size=float(0.2), random_state=0, print_results=True,
                          use_sgd_settings=False)
