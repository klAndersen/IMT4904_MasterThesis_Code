"""
Handles all operations related to reading and writing to file (including database retrieval)
"""

import pickle
from sklearn.externals.joblib import Memory
from os import listdir
from os.path import isfile, join
from mysqldatabase import MySQLDatabase
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from constants import QUESTION_TEXT_KEY, CLASS_LABEL_KEY, FILEPATH_TRAINING_DATA, FILEPATH_MODELS

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
def load_training_data(file_location=str, load_from_database=False, limit=int(1000),
                       clean_dataset=True, return_svmlight=False):
    """
    If ```load_from_database``` is True, retrieves and stores data from database to file.

    Arguments:
        file_location (str): Path + filename of libsvm file to save/load (e.g. 'training_data')
        load_from_database (bool): Should data be retrieved from database?
        limit (int): Amount of records to retrieve from database (default=1000)
        clean_dataset (bool): Should questions be cleaned (e.g. remove code samples, hexadecimals, numbers, etc)?
        return_svmlight (bool): Should ```sklearn.datasets.load_svmlight_file``` be returned?

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
        data = MySQLDatabase().retrieve_training_data(limit, clean_dataset)
        # create a term-document matrix
        vectorizer = CountVectorizer(analyzer='word', min_df=0.01, stop_words="english")
        td_matrix = vectorizer.fit_transform(data.get(QUESTION_TEXT_KEY))
        data.to_csv(csv_file)
        dump_svmlight_file(td_matrix, data[CLASS_LABEL_KEY], f=svm_file, comment=comment)
    if not return_svmlight:
        return DataFrame.from_csv(csv_file)
    return DataFrame.from_csv(csv_file), load_svmlight_file(svm_file)


def load_classifier_model_and_dataframe(model_name=str, dataset_file=str, limit=int(1000),
                                        load_from_database=False, return_svmlight=False):
    """
    Loads classifier model and pandas.DataFrame from file

    Arguments:
        model_name (str): Name of model to retrieve
        dataset_file (str): Name of file containing dataset
        limit (int): Amount of records to retrieve from database (default=1000)
        load_from_database (bool): Should data be retrieved from database?
        return_svmlight (bool): Should ```sklearn.datasets.load_svmlight_file``` be returned?

    Returns:
        tuple: pandas.DataFrame and loaded pickle model || None

    """
    model = get_training_model(FILEPATH_MODELS, model_name)
    so_dataframe = load_training_data(dataset_file, load_from_database, limit, True, return_svmlight)
    return so_dataframe, model
