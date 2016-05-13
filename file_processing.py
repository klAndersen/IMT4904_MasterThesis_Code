"""
Handles all operations related to reading and writing to file (including database retrieval)
"""

import pickle
import constants
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


def __get_training_model(path=str, model_name=str, suffix=".pkl"):
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
def __load_training_data(file_location=str, load_from_database=False, limit=int(1000), return_svmlight=False,
                         create_feature_detectors=False, create_unprocessed=False):
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
        comment = u"label: (-1: Bad question, +1: Good question); features: (term_id, frequency)"
        MySQLDatabase().set_vote_value_params()
        data = MySQLDatabase().retrieve_training_data(limit, create_feature_detectors, create_unprocessed)
        # create a term-document matrix
        vectorizer = CountVectorizer(analyzer='word', min_df=0.01, stop_words="english")
        td_matrix = vectorizer.fit_transform(data.get(constants.QUESTION_TEXT_KEY))
        data.to_csv(csv_file)
        dump_svmlight_file(td_matrix, data[constants.CLASS_LABEL_KEY], f=svm_file, comment=comment)
    if not return_svmlight:
        return DataFrame.from_csv(csv_file)
    return DataFrame.from_csv(csv_file), load_svmlight_file(svm_file)


def load_classifier_model_and_dataframe(model_name=str, dataset_file=str, limit=int(1000),
                                        load_from_database=False, return_svmlight=False,
                                        create_feature_detectors=False, create_unprocessed=False):
    """
    Loads classifier model and pandas.DataFrame from file

    Arguments:
        model_name (str): Name of model to retrieve
        dataset_file (str): Name of file containing dataset
        limit (int): Amount of records to retrieve from database (default=1000)
        load_from_database (bool): Should data be retrieved from database?
        return_svmlight (bool): Should ```sklearn.datasets.load_svmlight_file``` be returned?
        create_feature_detectors (bool): Is this function being called to create feature detectors?
        create_unprocessed (bool): Is this function being called to create a clean, unprocessed dataset?

    Returns:
        tuple: pandas.DataFrame and loaded pickle model || None

    """
    model = __get_training_model(constants.FILEPATH_MODELS, model_name)
    so_dataframe = __load_training_data(dataset_file, load_from_database, limit, return_svmlight,
                                        create_feature_detectors, create_unprocessed)
    return so_dataframe, model


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
    # convert DataFrame to list
    tag_list = tag_data[constants.TAG_NAME_COLUMN].tolist()
    return tag_list


def __create_and_save_feature_detectors(limit=int(1000)):
    """
    Creates feature detectors at the file location.

    The following feature detectors are created:
    - has_codeblock: Replaces code blocks with this text
    - has_link: Replaces links with this text
    - has_homework: Replaces "homework words" with this text
    - has_assignment: Replaces "assignment" with this text (homework scenario)
    - has_numeric: Replaces numerical values with this text
    - has_hexadecimal: Replaces hexadecimal values with this text

    Arguments:
        limit (int): Amount of rows to retrieve from database

    """
    file_location = constants.FILEPATH_FEATURE_DETECTOR + str(limit) + "_"
    __load_training_data(file_location, True, limit, False, True, False)


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
    __load_training_data(file_location, True, limit, False, False, True)

__create_unprocessed_dataset_dump(10000)
